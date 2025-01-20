use crate::calc_error::CalcError;
use crate::indexes::{Sender, Subscriber};
use ndarray::{ArrayView1, ArrayView2, Axis};
use std::sync::Arc;
#[derive(Default)]
pub struct PairsAndDistances;
impl PairsAndDistances {
    fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<(Vec<i8>, Vec<f64>), CalcError> {
        let n = y.len() * (y.len() - 1) / 2;
        let mut distances: Vec<f64> = Vec::with_capacity(n);
        let mut pairs_in_the_same_cluster: Vec<i8> = Vec::with_capacity(n);

        //calculating distances beetween pair of points and does they belong to the same cluster
        for (i, (row1, cluster1)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
            for (j, (row2, cluster2)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
                if i < j {
                    pairs_in_the_same_cluster.push((cluster1 != cluster2) as i8); // the same cluster =producer 0, different = 1
                    distances.push((&row2 - &row1).pow2().sum().sqrt());
                }
            }
        }
        Ok((pairs_in_the_same_cluster, distances))
    }
}

pub struct PairsAndDistancesNode<'a> {
    index: PairsAndDistances,
    sender: Sender<'a, (Vec<i8>, Vec<f64>)>,
}
impl<'a> PairsAndDistancesNode<'a> {
    pub fn new(sender: Sender<'a, (Vec<i8>, Vec<f64>)>) -> Self {
        Self {
            index: PairsAndDistances,
            sender,
        }
    }
}
impl<'a> Subscriber<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)>
    for PairsAndDistancesNode<'a>
{
    fn recieve_data(
        &mut self,
        data: Arc<Result<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>), CalcError>>,
    ) {
        let res = match data.as_ref() {
            Ok((x, y)) => self.index.compute(x, y),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(Arc::new(res));
    }
}
