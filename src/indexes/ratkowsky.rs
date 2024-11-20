use super::*;

pub struct Index {}
impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let x_mean = x
            .mean_axis(Axis(0))
            .ok_or("Cant compute mean for dataset")?;
        let clusters = calc_clusters(&y);
        let clusters_centroids = calc_clusters_centers(&clusters, &x);

        let (num_of_elems, num_of_vars) = x.dim();

        let mut bgss: Array1<f64> = Array1::zeros(num_of_vars);
        for (i, c) in clusters_centroids {
            bgss = bgss + clusters[&i].len() as f64 * (c - &x_mean).pow2();
        }

        let tss = x.var_axis(Axis(0), 0.) * num_of_elems as f64;

        let s_squared = (bgss / tss).sum() / num_of_vars as f64;
        let value = (s_squared / clusters.keys().len() as f64).sqrt();
        Ok(value)
    }
}
