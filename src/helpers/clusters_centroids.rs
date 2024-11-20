use crate::{
    helpers::{clusters::ClustersType, raw_data::RawDataType},
    indexes::{Subscribee, Subscriber},
    *,
};

pub type ClustersCentroidsType = Arc<HashMap<i32, Array1<f64>>>;
#[derive(Default)]
pub struct ClustersCentroids;
impl ClustersCentroids {
    pub fn compute(
        &self,
        clusters_indexes: &HashMap<i32, Array1<usize>>,
        data: &ArrayView2<f64>,
    ) -> HashMap<i32, Array1<f64>> {
        clusters_indexes
            .into_par_iter()
            .map(|(c, arr)| {
                let mut sum: Array1<f64> = Array1::zeros(data.ncols());
                for i in arr {
                    sum += &data.row(*i);
                }
                (*c, sum / arr.len() as f64)
            })
            .collect::<HashMap<i32, Array1<f64>>>()
    }
}
#[derive(Default)]
pub struct ClustersCentroidsNode<'a> {
    index: ClustersCentroids,
    clusters: Option<ClustersType>,
    raw_data: Option<RawDataType<'a>>,
    subscribee: Subscribee<'a, ClustersCentroidsType>,
}
impl<'a> ClustersCentroidsNode<'a> {
    pub fn with_subscribee(subscribee: Subscribee<'a, ClustersCentroidsType>) -> Self {
        Self {
            index: ClustersCentroids,
            clusters: None,
            raw_data: None,
            subscribee,
        }
    }
    fn process_when_ready(&mut self) {
        if let (Some((x, _)), Some(clusters)) = (self.raw_data.as_ref(), self.clusters.as_ref()) {
            let res = Arc::new(self.index.compute(clusters, x));
            self.subscribee.send_to_subscribers(&res);
        }
    }
}
impl<'a> Subscriber<ClustersType> for ClustersCentroidsNode<'a> {
    fn recieve_data(&mut self, data: &ClustersType) {
        self.clusters = Some(data.clone());
        self.process_when_ready();
    }
}

impl<'a> Subscriber<RawDataType<'a>> for ClustersCentroidsNode<'a> {
    fn recieve_data(&mut self, data: &RawDataType<'a>) {
        self.raw_data = Some(*data);
        self.process_when_ready();
    }
}
