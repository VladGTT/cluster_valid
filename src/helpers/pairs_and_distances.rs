use crate::*;
use helpers::raw_data::RawDataType;
use indexes::{Subscribee, Subscriber};
use ndarray::Axis;
type PairsAndDistancesType = (Arc<Vec<i8>>, Arc<Vec<f64>>);
#[derive(Default)]
pub struct PairsAndDistances;
impl PairsAndDistances {
    fn compute(&self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> PairsAndDistancesType {
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
        (Arc::new(pairs_in_the_same_cluster), Arc::new(distances))
    }
}

#[derive(Default)]
pub struct PairsAndDistancesNode<'a> {
    index: PairsAndDistances,
    subscribee: Subscribee<'a, PairsAndDistancesType>,
}
impl<'a> Subscriber<RawDataType<'a>> for PairsAndDistancesNode<'a> {
    fn recieve_data(&mut self, data: &RawDataType<'a>) {
        let (x, y) = data;
        let res = self.index.compute(x, y);
        self.subscribee.send_to_subscribers(&res);
    }
}
