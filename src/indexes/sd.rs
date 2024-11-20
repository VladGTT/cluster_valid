use super::*;
use std::ops::AddAssign;

pub struct IndexScat {}
pub struct IndexDis {}
impl Computable for IndexScat {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let features_variances = x.var_axis(Axis(0), 0.);
        let clusters = calc_clusters(&y);
        let cluster_centers = calc_clusters_centers(&clusters, &x);

        let mut stor: Vec<f64> = Vec::with_capacity(clusters.keys().len());

        for (c, arr) in clusters {
            let mut row = Array1::zeros(x.ncols());
            for i in arr.iter() {
                row.add_assign(&(&x.row(*i) - &cluster_centers[&c]).powi(2));
            }
            row /= arr.len() as f64;
            stor.push(calc_vector_euclidean_length(&row.view()));
        }
        let S = Array1::from_vec(stor).mean().ok_or("Cant calculate mean")?
            / calc_vector_euclidean_length(&features_variances.view());

        Ok(S)
    }
}
impl Computable for IndexDis {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let mut d = 0.0;
        let mut d_max = f64::MIN;
        let mut d_min = f64::MAX;
        let clusters = calc_clusters(&y);
        let cluster_centroids = calc_clusters_centers(&clusters, &x);

        for (i, row1) in &cluster_centroids {
            let mut dist_acum = 0.0;
            for (j, row2) in &cluster_centroids {
                if i != j {
                    let dist = find_euclidean_distance(&row1.view(), &row2.view());
                    dist_acum += dist;
                    if i < j {
                        if dist > d_max {
                            d_max = dist;
                        }
                        if dist < d_min {
                            d_min = dist;
                        }
                    }
                }
            }
            if dist_acum != 0.0 {
                d += 1. / dist_acum;
            }
        }
        let value = d * d_max / d_min;
        Ok(value)
    }
}
