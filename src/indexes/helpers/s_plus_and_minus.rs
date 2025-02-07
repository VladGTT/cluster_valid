use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use ndarray::{ArcArray1, ArrayView1};
use std::iter::zip;

#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(
        &self,
        pairs_in_the_same_cluster: &ArrayView1<i8>,
        distances: &ArrayView1<f64>,
    ) -> Result<(usize, usize, usize), CalcError> {
        let (mut s_plus, mut s_minus, mut ties) = (0, 0, 0);

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        //belonging to the same cluster

        for (i, (d1, b1)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
                if i < j && (*b1 == 1 && *b2 == 0) {
                    if d1 < d2 {
                        s_plus += 1;
                    }
                    if d1 > d2 {
                        s_minus += 1;
                    }
                    if d1 == d2 {
                        ties += 1;
                    }
                }
            }
        }
        Ok((s_plus, s_minus, ties))
    }
}
#[derive(Default)]
pub struct SPlusAndMinusNode<'a> {
    index: Index,
    sender: Sender<'a, (usize, usize, usize)>,
}

impl<'a> SPlusAndMinusNode<'a> {
    pub fn new(sender: Sender<'a, (usize, usize, usize)>) -> Self {
        Self {
            index: Index,
            sender,
        }
    }
}
impl<'a> Subscriber<(ArcArray1<i8>, ArcArray1<f64>)> for SPlusAndMinusNode<'a> {
    fn recieve_data(&mut self, data: Result<(ArcArray1<i8>, ArcArray1<f64>), CalcError>) {
        let res = match data.as_ref() {
            Ok((p, d)) => self.index.compute(&p.view(), &d.view()),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
