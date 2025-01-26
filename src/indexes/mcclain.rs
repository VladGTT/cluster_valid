use crate::calc_error::CalcError;
use ndarray::{ArrayView1, ArrayView2};
use std::iter::zip;

use crate::sender::{Sender, Subscriber};

#[derive(Clone, Copy, Debug)]
pub struct McclainIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(&self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<f64, CalcError> {
        let (mut num_pairs_the_same_clust, mut num_pairs_dif_clust): (f64, f64) = (0., 0.);
        let (mut sum_dist_same_clust, mut sum_dist_dif_clust): (f64, f64) = (0.0, 0.0);
        for (i, (row1, clust1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, clust2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = (&row2 - &row1).pow2().sum().sqrt();
                    if clust1 == clust2 {
                        sum_dist_same_clust += dist;
                        num_pairs_the_same_clust += 1.;
                    } else {
                        sum_dist_dif_clust += dist;
                        num_pairs_dif_clust += 1.;
                    }
                }
            }
        }
        let value = (sum_dist_same_clust / num_pairs_the_same_clust)
            / (sum_dist_dif_clust / num_pairs_dif_clust);
        Ok(value)
    }
}
pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, McclainIndexValue>,
}
impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, McclainIndexValue>) -> Self {
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
                .map(|val| McclainIndexValue { val }),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
