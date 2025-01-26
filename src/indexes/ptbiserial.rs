use crate::calc_error::CalcError;
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::iter::zip;

use crate::sender::{Sender, Subscriber};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct PtbiserialIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(&self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<f64, CalcError> {
        let (mut num_pairs_the_same_clust, mut num_pairs_dif_clust): (usize, usize) = (0, 0);
        let (mut sum_dist_same_clust, mut sum_dist_dif_clust): (f64, f64) = (0.0, 0.0);
        let num_pairs_total = x.nrows() * (x.nrows() - 1) / 2;
        let mut distances: Array1<f64> = Array1::zeros(num_pairs_total);
        let mut ctr = 0;
        for (i, (row1, clust1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, clust2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = (&row2 - &row1).pow2().sum().sqrt();
                    distances[ctr] = dist;
                    if clust1 == clust2 {
                        sum_dist_same_clust += dist;
                        num_pairs_the_same_clust += 1;
                    } else {
                        sum_dist_dif_clust += dist;
                        num_pairs_dif_clust += 1;
                    }
                    ctr += 1;
                }
            }
        }
        let std = distances.std(0.);

        let (num_pairs_the_same_clust, num_pairs_dif_clust, num_pairs_total) = (
            num_pairs_the_same_clust as f64,
            num_pairs_dif_clust as f64,
            num_pairs_total as f64,
        );

        let value = (sum_dist_same_clust / num_pairs_the_same_clust
            - sum_dist_dif_clust / num_pairs_dif_clust)
            * (num_pairs_dif_clust * num_pairs_the_same_clust).sqrt()
            / num_pairs_total;

        Ok(value)
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
impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for Node<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        let res = match data.as_ref() {
            Ok((x, y)) => self
                .index
                .compute(x, y)
                .map(|val| PtbiserialIndexValue { val }),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
