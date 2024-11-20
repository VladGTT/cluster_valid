use super::*;

pub struct Index {}
impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let cluster_centroids = calc_clusters_centers(&calc_clusters(&y), &x);
        let (n, d) = x.dim();
        let mut diffs: Array2<f64> = Array2::zeros((n, d));
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            let temp = &cluster_centroids[y] - &x;
            diffs.row_mut(i).assign(&temp);
        }

        let value = diffs.t().dot(&diffs).diag().sum();
        Ok(value)
    }
}
