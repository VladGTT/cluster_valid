use std::iter::zip;

use crate::calc_error::CalcError;
use ndarray::{ArcArray1, ArrayView1};

use crate::sender::{Sender, Subscriber};

#[derive(Clone, Copy, Debug)]
pub struct PtbiserialIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(
        &self,
        pairs_in_the_same_cluster: &ArrayView1<i8>,
        distances: &ArrayView1<f64>,
    ) -> Result<f64, CalcError> {
        let nt = pairs_in_the_same_cluster.len() as f64;
        let nw = pairs_in_the_same_cluster
            .iter()
            .filter(|i| **i == 1)
            .count() as f64;
        let sw = zip(pairs_in_the_same_cluster, distances)
            .filter(|(p, _)| **p == 1)
            .map(|(_, d)| *d)
            .sum::<f64>();
        let nb = pairs_in_the_same_cluster
            .iter()
            .filter(|i| **i == 0)
            .count() as f64;
        let sb = zip(pairs_in_the_same_cluster, distances)
            .filter(|(p, _)| **p == 0)
            .map(|(_, d)| *d)
            .sum::<f64>();
        // let std_d = distances.std(0.);
        // let val = ((sb / nb - sw / nw) * (nw * nb / (nt * nt)).sqrt()) / std_d;
        let val = ((sw / nw - sb / nb) * (nw * nb).sqrt()) / nt;
        Ok(val)
    }
}
pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, PtbiserialIndexValue>,
}
impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, PtbiserialIndexValue>) -> Self {
        Self {
            index: Index,
            sender,
        }
    }
}
impl<'a> Subscriber<(ArcArray1<i8>, ArcArray1<f64>)> for Node<'a> {
    fn recieve_data(&mut self, data: Result<(ArcArray1<i8>, ArcArray1<f64>), CalcError>) {
        let res = match data.as_ref() {
            Ok((p, d)) => self
                .index
                .compute(&p.view(), &d.view())
                .map(|val| PtbiserialIndexValue { val }),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
