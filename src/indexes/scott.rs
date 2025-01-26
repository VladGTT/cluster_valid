use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Determinant;
use std::{collections::HashMap, iter::zip, sync::Arc};

use crate::sender::{Sender, Subscriber};

#[derive(Clone, Copy, Debug)]
pub struct ScottIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        clusters_centroids: &HashMap<i32, Array1<f64>>,
    ) -> Result<f64, CalcError> {
        let x_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
        let mut diffs1: Array2<f64> = Array2::zeros(x.dim());
        let mut diffs2: Array2<f64> = Array2::zeros(x.dim());
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            diffs1.row_mut(i).assign(&(&x - &clusters_centroids[y]));
            diffs2.row_mut(i).assign(&(&x - &x_mean));
        }

        let w_q = diffs1.t().dot(&diffs1);
        let t = diffs2.t().dot(&diffs2);
        let det_t = Determinant::det(&t).map_err(|e| CalcError::from(format!("{e:?}")))?;
        let det_w_q = Determinant::det(&w_q).map_err(|e| CalcError::from(format!("{e:?}")))?;

        let val = x.nrows() as f64 * (det_t / det_w_q).ln();

        Ok(val)
    }
}

pub struct Node<'a> {
    index: Index,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    clusters_centroids: Option<Result<Arc<HashMap<i32, Array1<f64>>>, CalcError>>,
    sender: Sender<'a, ScottIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, ScottIndexValue>) -> Self {
        Self {
            index: Index,
            clusters_centroids: None,
            raw_data: None,
            sender,
        }
    }
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(clusters_centroids)) =
            (self.raw_data.as_ref(), self.clusters_centroids.as_ref())
        {
            let res = match raw_data.combine(clusters_centroids) {
                Ok(((x, y), cls_ctrds)) => self
                    .index
                    .compute(x, y, cls_ctrds)
                    .map(|val| ScottIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.clusters_centroids = None;
        }
    }
}

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for Node<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        self.raw_data = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<Arc<HashMap<i32, Array1<f64>>>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<Arc<HashMap<i32, Array1<f64>>>, CalcError>) {
        self.clusters_centroids = Some(data);
        self.process_when_ready();
    }
}
