use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{ArcArray1, ArcArray2, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Determinant;
use std::{collections::HashMap, sync::Arc};

use crate::sender::{Sender, Subscriber};

use super::helpers::{total_dispercion::TDValue, within_group_dispercion::WGDValue};

#[derive(Clone, Copy, Debug)]
pub struct ScottIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(
        &self,
        wg: &ArrayView2<f64>,
        td: &ArrayView2<f64>,
        counts: &ArrayView1<usize>,
    ) -> Result<f64, CalcError> {
        let n = counts.sum() as f64;
        let det_t = td.det().map_err(|e| CalcError::from(format!("{e:?}")))?;
        let det_wg = wg.det().map_err(|e| CalcError::from(format!("{e:?}")))?;
        let val = (det_t / det_wg).ln();
        let val = val * n;
        // let x_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
        // let mut diffs1: Array2<f64> = Array2::zeros(x.dim());
        // let mut diffs2: Array2<f64> = Array2::zeros(x.dim());
        // for (i, (x, y)) in zip(x.rows(), y).enumerate() {
        //     diffs1.row_mut(i).assign(&(&x - &clusters_centroids[y]));
        //     diffs2.row_mut(i).assign(&(&x - &x_mean));
        // }
        //
        // let w_q = diffs1.t().dot(&diffs1);
        // let t = diffs2.t().dot(&diffs2);
        // let det_t = Determinant::det(&t).map_err(|e| CalcError::from(format!("{e:?}")))?;
        // let det_w_q = Determinant::det(&w_q).map_err(|e| CalcError::from(format!("{e:?}")))?;
        //
        // let val = x.nrows() as f64 * (det_t / det_w_q).ln();
        //
        Ok(val)
    }
}

pub struct Node<'a> {
    index: Index,
    counts: Option<Result<ArcArray1<usize>, CalcError>>,
    wg: Option<Result<ArcArray2<f64>, CalcError>>,
    td: Option<Result<ArcArray2<f64>, CalcError>>,
    sender: Sender<'a, ScottIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, ScottIndexValue>) -> Self {
        Self {
            index: Index,
            wg: None,
            td: None,
            counts: None,
            sender,
        }
    }
    fn process_when_ready(&mut self) {
        if let (Some(wg), Some(td), Some(counts)) =
            (self.wg.as_ref(), self.td.as_ref(), self.counts.as_ref())
        {
            let res = match wg.combine(td).combine(counts) {
                Ok(((wg, td), cnts)) => self
                    .index
                    .compute(&wg.view(), &td.view(), &cnts.view())
                    .map(|val| ScottIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.wg = None;
            self.td = None;
            self.counts = None;
        }
    }
}
impl<'a> Subscriber<ArcArray1<usize>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<ArcArray1<usize>, CalcError>) {
        self.counts = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<WGDValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<WGDValue, CalcError>) {
        self.wg = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}

impl<'a> Subscriber<TDValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<TDValue, CalcError>) {
        self.td = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}
