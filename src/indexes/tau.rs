use crate::calc_error::{CalcError, CombineErrors};
use crate::sender::{Sender, Subscriber};
use core::f64;
use ndarray::{ArrayView1, ArrayView2};
use std::{iter::zip, sync::Arc};

#[derive(Clone, Copy, Debug)]
pub struct TauIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        y: &ArrayView1<i32>,
        pairs_and_distances: &(Vec<i8>, Vec<f64>),
    ) -> Result<f64, CalcError> {
        let (pairs_in_the_same_cluster, distances) = pairs_and_distances;
        let total_number_of_pairs = pairs_in_the_same_cluster.len();
        let (mut s_plus, mut s_minus): (usize, usize) = (0, 0);

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        // belonging to the same cluster
        for (i, (d1, b1)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
                if i < j && *b1 == 0 && *b2 == 1 {
                    s_plus += (d1 < d2) as u8 as usize;
                    s_minus += (d1 > d2) as u8 as usize;
                }
            }
        }
        let nw: usize = pairs_in_the_same_cluster
            .iter()
            .filter(|i| **i == 1)
            .map(|f| *f as usize)
            .sum();
        let nb: usize = total_number_of_pairs - nw;
        let v0 = (total_number_of_pairs * (total_number_of_pairs - 1)) as f64 / 2.0;
        // let value = (s_plus - s_minus) as f64 / ((v0 - t as f64) * v0).sqrt();
        let value = (s_plus - s_minus) as f64 / (nb as f64 * nw as f64 * v0).sqrt();
        Ok(value)
    }
}

pub struct Node<'a> {
    index: Index,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    pairs_and_distances: Option<Result<Arc<(Vec<i8>, Vec<f64>)>, CalcError>>,
    sender: Sender<'a, TauIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(pairs_and_distances)) =
            (self.raw_data.as_ref(), self.pairs_and_distances.as_ref())
        {
            let res = match raw_data.combine(pairs_and_distances) {
                Ok(((_, y), pairs_and_distances)) => self
                    .index
                    .compute(y, pairs_and_distances)
                    .map(|val| TauIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.pairs_and_distances = None;
        }
    }
    pub fn new(sender: Sender<'a, TauIndexValue>) -> Self {
        Self {
            index: Index,
            raw_data: None,
            pairs_and_distances: None,
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
impl<'a> Subscriber<Arc<(Vec<i8>, Vec<f64>)>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<Arc<(Vec<i8>, Vec<f64>)>, CalcError>) {
        self.pairs_and_distances = Some(data);
        self.process_when_ready();
    }
}
