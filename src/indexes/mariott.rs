use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{ArcArray1, ArcArray2, ArrayView1, ArrayView2};
use ndarray_linalg::Determinant;

use crate::sender::{Sender, Subscriber};

use super::helpers::within_group_dispercion::WGDValue;
#[derive(Clone, Copy, Debug)]
pub struct MariottIndexValue {
    pub val: f64,
}

#[derive(Default)]
pub struct Index;
impl Index {
    pub fn compute(
        &self,
        counts: &ArrayView1<usize>,
        wg: &ArrayView2<f64>,
    ) -> Result<f64, CalcError> {
        let q = counts.len();
        let q2 = (q * q) as f64;
        let det_wg = wg.det().map_err(|e| CalcError::from(format!("{e:?}")))?;
        let val = q2 * det_wg;
        // let mut diffs: Array2<f64> = Array2::zeros(x.dim());
        // for (i, (x, y)) in zip(x.rows(), y).enumerate() {
        //     diffs.row_mut(i).assign(&(&x - &clusters_centroids[y]));
        // }
        //
        // let w_q: Array2<f64> = diffs.t().dot(&diffs);
        // let det_w_q = Determinant::det(&w_q).map_err(|e| CalcError::from(format!("{e:?}")))?;
        // let q = clusters_centroids.keys().len() as f64;
        // let val = q.powi(2) * det_w_q;
        Ok(val)
    }
}

pub struct Node<'a> {
    index: Index,
    counts: Option<Result<ArcArray1<usize>, CalcError>>,
    wg: Option<Result<ArcArray2<f64>, CalcError>>,
    sender: Sender<'a, MariottIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, MariottIndexValue>) -> Self {
        Self {
            index: Index,
            counts: None,
            wg: None,
            sender,
        }
    }
    fn process_when_ready(&mut self) {
        if let (Some(counts), Some(wg)) = (self.counts.as_ref(), self.wg.as_ref()) {
            let res = match wg.combine(counts) {
                Ok((wg, cnts)) => self
                    .index
                    .compute(&cnts.view(), &wg.view())
                    .map(|val| MariottIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.wg = None;
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
