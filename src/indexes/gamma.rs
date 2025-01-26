use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use std::iter::zip;
use std::sync::Arc;
#[derive(Clone, Copy, Debug)]
pub struct GammaIndexValue {
    pub val: f64,
}

#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(&self, pairs_and_distances: &(Vec<i8>, Vec<f64>)) -> Result<f64, CalcError> {
        let (pairs_in_the_same_cluster, distances) = pairs_and_distances;
        let (mut s_plus, mut s_minus) = (0.0, 0.0);

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        //belonging to the same cluster

        for (i, (d1, b1)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
                if i < j && (*b1 == 1 && *b2 == 0) {
                    if d1 < d2 {
                        s_plus += 1.0;
                    }
                    if d1 > d2 {
                        s_minus += 1.0;
                    }
                }
            }
        }
        let value = (s_plus - s_minus) / (s_plus + s_minus);
        Ok(value)
    }
}
#[derive(Default)]
pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, GammaIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, GammaIndexValue>) -> Self {
        Self {
            index: Index,
            sender,
        }
    }
}
impl<'a> Subscriber<Arc<(Vec<i8>, Vec<f64>)>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<Arc<(Vec<i8>, Vec<f64>)>, CalcError>) {
        let res = match data.as_ref() {
            Ok(pairs_and_distances) => self
                .index
                .compute(pairs_and_distances)
                .map(|val| GammaIndexValue { val }),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
