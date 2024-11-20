use super::*;

pub struct Index {}
impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let cluster_centroids = calc_clusters_centers(&calc_clusters(&y), &x);
        let x_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
        let mut diffs1: Array2<f64> = Array2::zeros(x.dim());
        let mut diffs2: Array2<f64> = Array2::zeros(x.dim());
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            diffs1.row_mut(i).assign(&(&x - &cluster_centroids[y]));
            diffs2.row_mut(i).assign(&(&x - &x_mean));
        }

        let w_q = diffs1.t().dot(&diffs1);
        let t = diffs2.t().dot(&diffs2);
        let det_t = calc_matrix_determinant(&t.view())?;
        let det_w_q = calc_matrix_determinant(&w_q.view())?;
        let value = det_t / det_w_q;

        Ok(value)
    }
}
