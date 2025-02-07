use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use itertools::*;
use ndarray::{ArrayView1, ArrayView2};
use std::sync::Arc;
#[derive(Default)]
pub struct Clusters;
impl Clusters {
    pub fn compute(&self, cluster_indexes: &ArrayView1<i32>) -> Result<Vec<Vec<usize>>, CalcError> {
        // let clusters = cluster_indexes
        //     .iter()
        //     .enumerate()
        //     .map(|(i, c)| (*c, i))
        //     .into_group_map();
        // let res = clusters
        //     .into_iter()
        //     .map(|(i, val)| (i, Array1::from_vec(val)))
        //     .collect();
        let res = cluster_indexes
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, i))
            .into_group_map()
            .into_iter()
            .sorted_by(|(a, _), (b, _)| a.cmp(b))
            .map(|(_, v)| v)
            .collect::<Vec<Vec<usize>>>();
        Ok(res)
    }
}
pub struct ClustersNode<'a> {
    index: Clusters,
    sender: Sender<'a, Arc<Vec<Vec<usize>>>>,
}
impl<'a> ClustersNode<'a> {
    pub fn new(sender: Sender<'a, Arc<Vec<Vec<usize>>>>) -> Self {
        Self {
            index: Clusters,
            sender,
        }
    }
}
impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for ClustersNode<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        let res = match data.as_ref() {
            Ok((_, y)) => self.index.compute(y).map(|v| Arc::new(v)),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
