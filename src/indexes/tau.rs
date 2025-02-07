use crate::calc_error::{CalcError, CombineErrors};
use crate::sender::{Sender, Subscriber};
use core::f64;
use ndarray::{ArcArray1, ArrayView1};

#[derive(Clone, Copy, Debug)]
pub struct TauIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        pairs_in_the_same_cluster: &ArrayView1<i8>,
        s_plus: usize,
        s_minus: usize,
        ties: usize,
    ) -> Result<f64, CalcError> {
        let nt = pairs_in_the_same_cluster.len() as f64;
        let temp = nt * (nt - 1.) / 2.;
        let value = (s_plus - s_minus) as f64 / (temp * (temp - ties as f64)).sqrt();
        Ok(value)

        // let nw = pairs_in_the_same_cluster
        //     .iter()
        //     .filter(|i| **i == 1)
        //     .count() as f64;
        // let nb = nt - nw;
        // let value = (s_plus - s_minus) as f64 / (temp * nw * nb).sqrt();
        // Ok(value)

        // return Err(CalcError::from(format!("{pairs_and_distances:?}")));
        // let (pairs_in_the_same_cluster, distances) = pairs_and_distances;
        // let total_number_of_pairs = pairs_in_the_same_cluster.len();
        // let (mut s_plus, mut s_minus): (usize, usize) = (0, 0);
        //
        // // finding s_plus which represents the number of times a distance between two points
        // // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        // // belonging to the same cluster
        //
        // for (d1, _) in zip(distances, pairs_in_the_same_cluster).filter(|(_, p)| **p == 0) {
        //     let mut is_smaller = true;
        //     for (d2, _) in zip(distances, pairs_in_the_same_cluster).filter(|(_, p)| **p == 1) {
        //         is_smaller &= d2 < d1;
        //     }
        //     s_plus += is_smaller as usize;
        //     s_minus += !is_smaller as usize;
        // }
        // // return Err(CalcError::from(format!("s+ {s_plus} s- {s_minus}")));
        // let nw: usize = pairs_in_the_same_cluster
        //     .iter()
        //     .filter(|i| **i == 1)
        //     .map(|f| *f as usize)
        //     .sum();
        // let nb: usize = total_number_of_pairs - nw;
        // let v0 = (total_number_of_pairs * (total_number_of_pairs - 1)) as f64 / 2.0;
        // // let value = (s_plus - s_minus) as f64 / ((v0 - t as f64) * v0).sqrt();
        // let value = (s_plus - s_minus) as f64 / (nb as f64 * nw as f64 * v0).sqrt();
        // Ok(value)
    }
}

pub struct Node<'a> {
    index: Index,
    s_plus_and_minus: Option<Result<(usize, usize, usize), CalcError>>,
    pairs_and_distances: Option<Result<(ArcArray1<i8>, ArcArray1<f64>), CalcError>>,
    sender: Sender<'a, TauIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(s_plus_and_minus), Some(pairs_and_distances)) = (
            self.s_plus_and_minus.as_ref(),
            self.pairs_and_distances.as_ref(),
        ) {
            let res = match s_plus_and_minus.combine(pairs_and_distances) {
                Ok(((s_plus, s_minus, ties), (pairs, _))) => self
                    .index
                    .compute(&pairs.view(), *s_plus, *s_minus, *ties)
                    .map(|val| TauIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.s_plus_and_minus = None;
            self.pairs_and_distances = None;
        }
    }
    pub fn new(sender: Sender<'a, TauIndexValue>) -> Self {
        Self {
            index: Index,
            s_plus_and_minus: None,
            pairs_and_distances: None,
            sender,
        }
    }
}

impl<'a> Subscriber<(usize, usize, usize)> for Node<'a> {
    fn recieve_data(&mut self, data: Result<(usize, usize, usize), CalcError>) {
        self.s_plus_and_minus = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<(ArcArray1<i8>, ArcArray1<f64>)> for Node<'a> {
    fn recieve_data(&mut self, data: Result<(ArcArray1<i8>, ArcArray1<f64>), CalcError>) {
        self.pairs_and_distances = Some(data);
        self.process_when_ready();
    }
}
