use super::*;

pub struct Index {}
impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let cluster_centroids = calc_clusters_centers(&calc_clusters(&y), &x);
        let mut diffs: Array2<f64> = Array2::zeros((x.nrows(), x.ncols()));
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            let temp = &cluster_centroids[y] - &x;
            diffs.row_mut(i).assign(&temp);
        }
        let w = diffs.t().dot(&diffs);

        let w_norm = &w - &w.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
        let n = w.nrows() as f64;
        let value = (w_norm.t().dot(&w_norm) / n).diag().sum();
        Ok(value)
    }
}
