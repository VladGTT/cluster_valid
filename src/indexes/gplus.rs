use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{ArrayView1, ArrayView2};
use std::{iter::zip, sync::Arc};

use crate::sender::{Sender, Subscriber};

#[derive(Clone, Copy, Debug)]
pub struct GplusIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(
        &self,
        x: &ArrayView2<f64>,
        pairs_and_distances: &(Vec<i8>, Vec<f64>),
    ) -> Result<f64, CalcError> {
        let (pairs_in_the_same_cluster, distances) = pairs_and_distances;
        let mut s_minus = 0.0;

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        //belonging to the same cluster

        for (i, (d1, b1)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
                if i < j && (*b1 == 0 && *b2 == 1) && d1 > d2 {
                    s_minus += 1.0;
                }
            }
        }

        let (n, _) = x.dim();
        let n_t = n as f64 * (n as f64 - 1.) / 2.0;
        let value = 2. * s_minus / (n_t * (n_t - 1.0));
        Ok(value)
    }
}

pub struct Node<'a> {
    index: Index,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    pairs_and_distances: Option<Result<Arc<(Vec<i8>, Vec<f64>)>, CalcError>>,
    sender: Sender<'a, GplusIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(pairs_and_distances)) =
            (self.raw_data.as_ref(), self.pairs_and_distances.as_ref())
        {
            let res = match raw_data.combine(pairs_and_distances) {
                Ok(((x, _), pairs_and_distances)) => self
                    .index
                    .compute(x, pairs_and_distances)
                    .map(|val| GplusIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.pairs_and_distances = None;
        }
    }
    pub fn new(sender: Sender<'a, GplusIndexValue>) -> Self {
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
