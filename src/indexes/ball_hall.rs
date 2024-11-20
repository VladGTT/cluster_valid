use super::*;
use helpers::{
    clusters::ClustersType, clusters_centroids::ClustersCentroidsType, raw_data::RawDataType,
};
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(
        &self,
        x: &ArrayView2<f64>,
        clusters_centroids: &HashMap<i32, Array1<f64>>,
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> f64 {
        let std = clusters
            .par_iter()
            .map(|(c, arr)| {
                arr.iter()
                    .map(|i| (&x.row(*i) - &clusters_centroids[c]).pow2().sum())
                    .sum::<f64>()
                    / arr.len() as f64
            })
            .sum::<f64>();

        std / (clusters.keys().len() as f64)
    }
}

#[derive(Default)]
pub struct Node<'a> {
    index: Index,
    raw_data: Option<RawDataType<'a>>,
    clusters: Option<ClustersType>,
    clusters_centroids: Option<ClustersCentroidsType>,
    pub res: Option<f64>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some((x, _)), Some(clusters), Some(clusters_centroids)) = (
            self.raw_data.as_ref(),
            self.clusters.as_ref(),
            self.clusters_centroids.as_ref(),
        ) {
            self.res = Some(self.index.compute(x, clusters_centroids, clusters));
        }
    }
}

impl<'a> Subscriber<RawDataType<'a>> for Node<'a> {
    fn recieve_data(&mut self, data: &RawDataType<'a>) {
        self.raw_data = Some(*data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<ClustersType> for Node<'a> {
    fn recieve_data(&mut self, data: &ClustersType) {
        self.clusters = Some(data.clone());
        self.process_when_ready();
    }
}
impl<'a> Subscriber<ClustersCentroidsType> for Node<'a> {
    fn recieve_data(&mut self, data: &ClustersCentroidsType) {
        self.clusters_centroids = Some(data.clone());
        self.process_when_ready();
    }
}
