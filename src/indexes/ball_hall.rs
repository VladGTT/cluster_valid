use crate::calc_error::{CalcError, CombineErrors};
use crate::sender::{Sender, Subscriber};
use ndarray::{ArcArray1, ArcArray2, ArrayView1, ArrayView2};

use super::helpers::within_group_dispercion::WGDValue;
#[derive(Clone, Copy, Debug)]
pub struct BallHallIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(&self, wg: &ArrayView2<f64>, cnts: &ArrayView1<usize>) -> Result<f64, CalcError> {
        let trace_wg = wg.diag().sum();
        let q = cnts.len();
        // let std = clusters
        //     .par_iter()
        //     .map(|(c, arr)| {
        //         arr.iter()
        //             .map(|i| (&x.row(*i) - &clusters_centroids[c]).pow2().sum())
        //             .sum::<f64>()
        //             / arr.len() as f64
        //     })
        //     .sum::<f64>();
        // let val = std / (clusters.keys().len() as f64);
        Ok(trace_wg / q as f64)
    }
}

pub struct Node<'a> {
    index: Index,
    wg: Option<Result<ArcArray2<f64>, CalcError>>,
    counts: Option<Result<ArcArray1<usize>, CalcError>>,
    sender: Sender<'a, BallHallIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(wg), Some(counts)) = (self.wg.as_ref(), self.counts.as_ref()) {
            let res = match wg.combine(counts) {
                Ok((wg, cnts)) => self
                    .index
                    .compute(&wg.view(), &cnts.view())
                    .map(|val| BallHallIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.wg = None;
            self.counts = None;
        }
    }
    pub fn new(sender: Sender<'a, BallHallIndexValue>) -> Self {
        Self {
            index: Index,
            wg: None,
            counts: None,
            sender,
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
