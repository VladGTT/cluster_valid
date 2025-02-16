use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use ndarray::{ArcArray1, ArrayView1, ArrayView2, Axis};
#[derive(Default)]
pub struct PairsAndDistances;
impl PairsAndDistances {
    fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<(ArcArray1<i8>, ArcArray1<f64>), CalcError> {
        let n = y.len() * (y.len() - 1) / 2;
        let mut distances: Vec<f64> = Vec::with_capacity(n);
        let mut pairs_in_the_same_cluster: Vec<i8> = Vec::with_capacity(n);

        //calculating distances beetween pair of points and does they belong to the same cluster
        for (i, (row1, cluster1)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
            for (j, (row2, cluster2)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
                if i < j {
                    pairs_in_the_same_cluster.push((cluster1 == cluster2) as i8); // the same cluster =producer 1, different = 0
                    distances.push((&row2 - &row1).pow2().sum().sqrt());
                }
            }
        }
        let pairs_in_the_same_cluster = ArcArray1::from_vec(pairs_in_the_same_cluster);
        let distances = ArcArray1::from_vec(distances);
        Ok((pairs_in_the_same_cluster, distances))
    }
}

pub struct PairsAndDistancesNode<'a> {
    index: PairsAndDistances,
    sender: Sender<'a, (ArcArray1<i8>, ArcArray1<f64>)>,
}
impl<'a> PairsAndDistancesNode<'a> {
    pub fn new(sender: Sender<'a, (ArcArray1<i8>, ArcArray1<f64>)>) -> Self {
        Self {
            index: PairsAndDistances,
            sender,
        }
    }
}
impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for PairsAndDistancesNode<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        let res = match data.as_ref() {
            Ok((x, y)) => self.index.compute(x, y),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
