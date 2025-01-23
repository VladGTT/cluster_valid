use crate::calc_error::{CalcError, CombineErrors};
use core::f64;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Inverse;
use std::{collections::HashMap, iter::zip, sync::Arc};

use crate::sender::{Sender, Subscriber};

#[derive(Clone, Copy, Debug)]
pub struct FriedmanIndexValue {
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
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> Result<f64, CalcError> {
        let (n, d) = x.dim();
        let mut dif: Array2<f64> = Array2::zeros((n, d));
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            let temp = &clusters_centroids[y] - &x;
            dif.row_mut(i).assign(&temp);
        }
        let wg = dif.t().dot(&dif);
        let wg_inv = Inverse::inv(&wg).map_err(|e| e.to_string())?;

        let data_center = x.mean_axis(Axis(0)).ok_or("Cant calc data centroid")?;

        let mut b: Array2<f64> = Array2::zeros((n, d));

        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            let temp = &data_center - &clusters_centroids[y];
            b.row_mut(i).assign(&temp);
        }

        let bg = b.t().dot(&b);
        let value = wg_inv.dot(&bg).diag().sum();
        Ok(value)
    }
}
pub struct Node<'a> {
    index: Index,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    clusters: Option<Result<Arc<HashMap<i32, Array1<usize>>>, CalcError>>,
    clusters_centroids: Option<Result<Arc<HashMap<i32, Array1<f64>>>, CalcError>>,
    sender: Sender<'a, FriedmanIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(clusters), Some(clusters_centroids)) = (
            self.raw_data.as_ref(),
            self.clusters.as_ref(),
            self.clusters_centroids.as_ref(),
        ) {
            let res = match raw_data.combine(clusters).combine(clusters_centroids) {
                Ok((((x, y), cls), cls_ctrds)) => self
                    .index
                    .compute(x, y, cls_ctrds, cls)
                    .map(|val| FriedmanIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.clusters = None;
            self.clusters_centroids = None;
        }
    }
    pub fn new(sender: Sender<'a, FriedmanIndexValue>) -> Self {
        Self {
            index: Index,
            raw_data: None,
            clusters_centroids: None,
            clusters: None,
            sender,
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
impl<'a> Subscriber<Arc<HashMap<i32, Array1<usize>>>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<Arc<HashMap<i32, Array1<usize>>>, CalcError>) {
        self.clusters = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<Arc<HashMap<i32, Array1<f64>>>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<Arc<HashMap<i32, Array1<f64>>>, CalcError>) {
        self.clusters_centroids = Some(data);
        self.process_when_ready();
    }
}
