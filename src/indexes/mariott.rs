use super::*;
use helpers::{clusters_centroids::ClustersCentroidsType, raw_data::RawDataType};
use ndarray_linalg::Determinant;
#[derive(Default)]
pub struct Index;
impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        clusters_centroids: &HashMap<i32, Array1<f64>>,
    ) -> Result<f64, CalcError> {
        let mut diffs: Array2<f64> = Array2::zeros(x.dim());
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            diffs.row_mut(i).assign(&(&x - &clusters_centroids[y]));
        }

        let w_q: Array2<f64> = diffs.t().dot(&diffs);
        let det_w_q = Determinant::det(&w_q).map_err(|_| "Cant calc Determinant")?;
        let q = clusters_centroids.keys().len() as f64;
        let value = q.powi(2) * det_w_q;
        Ok(value)
    }
}

#[derive(Default)]
pub struct Node<'a> {
    index: Index,
    raw_data: Option<RawDataType<'a>>,
    clusters_centroids: Option<ClustersCentroidsType>,
    pub res: Option<Result<f64, CalcError>>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some((x, y)), Some(clusters_centroids)) =
            (self.raw_data.as_ref(), self.clusters_centroids.as_ref())
        {
            self.res = Some(self.index.compute(x, y, clusters_centroids));
        }
    }
}

impl<'a> Subscriber<RawDataType<'a>> for Node<'a> {
    fn recieve_data(&mut self, data: &RawDataType<'a>) {
        self.raw_data = Some(*data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<ClustersCentroidsType> for Node<'a> {
    fn recieve_data(&mut self, data: &ClustersCentroidsType) {
        self.clusters_centroids = Some(data.clone());
        self.process_when_ready();
    }
}
