use crate::{
    helpers::raw_data::RawDataType,
    indexes::{Subscribee, Subscriber},
    *,
};
use itertools::*;
use std::{ops::AddAssign, sync::Arc};
pub type ClustersType = Arc<HashMap<i32, Array1<usize>>>;
#[derive(Default)]
pub struct Clusters;
impl Clusters {
    pub fn compute(&self, cluster_indexes: &ArrayView1<i32>) -> HashMap<i32, Array1<usize>> {
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

        cluster_indexes_with_counter
            .into_iter()
            .map(|(i, (arr, _))| (i, arr))
            .collect::<HashMap<i32, Array1<usize>>>()
    }
}
#[derive(Default)]
pub struct ClustersNode<'a> {
    index: Clusters,
    subscribee: Subscribee<'a, ClustersType>,
}
impl<'a> ClustersNode<'a> {
    pub fn with_subscribee(subscribee: Subscribee<'a, ClustersType>) -> Self {
        Self {
            index: Clusters,
            subscribee,
        }
    }
}
impl<'a> Subscriber<RawDataType<'a>> for ClustersNode<'a> {
    fn recieve_data(&mut self, data: &RawDataType) {
        let (_, y) = data;
        let res = Arc::new(self.index.compute(y));
        self.subscribee.send_to_subscribers(&res);
    }
}
