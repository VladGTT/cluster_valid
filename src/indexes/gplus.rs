use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{ArcArray1, ArrayView1, ArrayView2};
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
        pairs_in_the_same_cluster: &ArrayView1<i8>,
        s_minus: usize,
    ) -> Result<f64, CalcError> {
        let nt = pairs_in_the_same_cluster.len() as f64;
        let value = 2. * s_minus as f64 / (nt * (nt - 1.0));
        Ok(value)

        // let (pairs_in_the_same_cluster, distances) = pairs_and_distances;
        // let mut s_minus = 0.0;
        //
        // // finding s_plus which represents the number of times a distance between two points
        // // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        // //belonging to the same cluster
        //
        // for (i, (d1, b1)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
        //     for (j, (d2, b2)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
        //         if i < j && (*b1 == 0 && *b2 == 1) && d1 > d2 {
        //             s_minus += 1.0;
        //         }
        //     }
        // }
        //
        // let (n, _) = x.dim();
        // let n_t = n as f64 * (n as f64 - 1.) / 2.0;
        // let value = 2. * s_minus / (n_t * (n_t - 1.0));
        // Ok(value)
    }
}

pub struct Node<'a> {
    index: Index,
    s_plus_and_minus: Option<Result<(usize, usize, usize), CalcError>>,
    pairs_and_distances: Option<Result<(ArcArray1<i8>, ArcArray1<f64>), CalcError>>,
    sender: Sender<'a, GplusIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(s_plus_and_minus), Some(pairs_and_distances)) = (
            self.s_plus_and_minus.as_ref(),
            self.pairs_and_distances.as_ref(),
        ) {
            let res = match s_plus_and_minus.combine(pairs_and_distances) {
                Ok(((_, s_minus, _), (pairs, _))) => self
                    .index
                    .compute(&pairs.view(), *s_minus)
                    .map(|val| GplusIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.s_plus_and_minus = None;
            self.pairs_and_distances = None;
        }
    }
    pub fn new(sender: Sender<'a, GplusIndexValue>) -> Self {
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
