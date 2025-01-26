use crate::calc_error::CalcError;
use ndarray::{ArrayView1, ArrayView2};
use std::iter::zip;

use crate::sender::{Sender, Subscriber};

#[derive(Clone, Copy, Debug)]
pub struct DunnIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    pub fn compute(&self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<f64, CalcError> {
        let n = y.len() * (y.len() - 1) / 2;
        let mut intercluster_distances: Vec<f64> = Vec::with_capacity(n);
        let mut intracluster_distances: Vec<f64> = Vec::with_capacity(n);

        for (i, (row1, cluster1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, cluster2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = (&row2 - &row1).pow2().sum().sqrt();
                    if cluster1 == cluster2 {
                        intracluster_distances.push(dist);
                    } else {
                        intercluster_distances.push(dist);
                    }
                }
            }
        }

        let max_intracluster = intracluster_distances
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .ok_or("Can't find max intracluster distance")?;
        let min_intercluster = intercluster_distances
            .iter()
            .min_by(|x, y| x.total_cmp(y))
            .ok_or("Can't find min intercluster distance")?;

        let val = min_intercluster / max_intracluster;
        Ok(val)
    }
}

pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, DunnIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, DunnIndexValue>) -> Self {
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
            Ok((x, y)) => self.index.compute(x, y).map(|val| DunnIndexValue { val }),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
