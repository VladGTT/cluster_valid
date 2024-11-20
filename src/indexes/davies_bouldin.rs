use super::*;
use calc_error::CalcError;
use helpers::{
    clusters::ClustersType, clusters_centroids::ClustersCentroidsType, raw_data::RawDataType,
};
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        clusters: &HashMap<i32, Array1<usize>>,
        clusters_centroids: &HashMap<i32, Array1<f64>>,
    ) -> Result<f64, CalcError> {
        let mut stor: HashMap<i32, f64> = HashMap::default();

        for (c, arr) in clusters.iter() {
            let temp = arr
                .par_iter()
                .map(|i| (&x.row(*i) - &clusters_centroids[c]).pow2().sum().sqrt())
                .sum::<f64>()
                / arr.len() as f64;
            stor.insert(*c, temp);
        }
        let q = clusters.keys().len();

        let mut acum = 0.0;
        let mut temp: Vec<f64> = Vec::with_capacity(q);
        for i in clusters.keys() {
            for j in clusters.keys() {
                if *i != *j {
                    let coef = (stor[i] + stor[j])
                        / (&clusters_centroids[j] - &clusters_centroids[i])
                            .pow2()
                            .sum()
                            .sqrt();
                    temp.push(coef);
                }
            }
            acum += temp
                .iter()
                .max_by(|a, b| a.total_cmp(b))
                .ok_or("Cant find max val")?;
            temp.clear();
        }

        let value = acum / q as f64;
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
            self.res = Some(self.index.compute(x, clusters, clusters_centroids));
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
