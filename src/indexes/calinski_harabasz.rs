use core::f64;

use super::*;
use calc_error::CalcError;
use helpers::{
    clusters::ClustersType, clusters_centroids::ClustersCentroidsType, raw_data::RawDataType,
};
use ndarray::Axis;
#[derive(Default)]
pub struct Index {}
impl Index {
    fn compute(
        &self,
        x: &ArrayView2<f64>,

        clusters_centroids: &HashMap<i32, Array1<f64>>,
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> Result<f64, CalcError> {
        let number_of_objects = x.nrows() as f64;
        let data_center = x.mean_axis(Axis(0)).ok_or("Cant calc data centroid")?;
        let number_of_clusters = clusters.keys().len() as f64;

        let inbetween_group_dispersion = clusters_centroids
            .iter()
            .map(|(i, c)| (&data_center - c).pow2().sum() * clusters[i].len() as f64)
            .sum::<f64>();

        let within_group_dispersion = clusters
            .iter()
            .map(|(c, arr)| {
                arr.iter()
                    .map(|i| (&x.row(*i) - &clusters_centroids[c]).pow2().sum())
                    .sum::<f64>()
            })
            .sum::<f64>();
        let value = (inbetween_group_dispersion / (number_of_clusters - 1.0))
            / (within_group_dispersion / (number_of_objects - number_of_clusters));

        Ok(value)
    }
}
#[derive(Default)]
pub struct Node<'a> {
    index: Index,
    raw_data: Option<RawDataType<'a>>,
    clusters: Option<ClustersType>,
    clusters_centroids: Option<ClustersCentroidsType>,
    pub res: Option<Result<f64, CalcError>>,
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
