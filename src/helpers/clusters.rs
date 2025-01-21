use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use itertools::*;
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::{collections::HashMap, ops::AddAssign, sync::Arc};
#[derive(Default)]
pub struct Clusters;
impl Clusters {
    pub fn compute(
        &self,
        cluster_indexes: &ArrayView1<i32>,
    ) -> Result<HashMap<i32, Array1<usize>>, CalcError> {
        let mut cluster_indexes_with_counter = cluster_indexes
            .iter()
            .counts()
            .into_iter()
            .map(|(i, n)| (*i, (Array1::zeros(n), 0)))
            .collect::<HashMap<i32, (Array1<usize>, usize)>>();
        for (i, c) in cluster_indexes.iter().enumerate() {
            let (arr, ctr) = cluster_indexes_with_counter.get_mut(c).unwrap();
            arr[*ctr] = i;
            ctr.add_assign(1);
        }

        let res = cluster_indexes_with_counter
            .into_iter()
            .map(|(i, (arr, _))| (i, Array1::from(arr)))
            .collect::<HashMap<i32, Array1<usize>>>();
        Ok(res)
    }
}
pub struct ClustersNode<'a> {
    index: Clusters,
    sender: Sender<'a, Arc<HashMap<i32, Array1<usize>>>>,
}
impl<'a> ClustersNode<'a> {
    pub fn new(sender: Sender<'a, Arc<HashMap<i32, Array1<usize>>>>) -> Self {
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
