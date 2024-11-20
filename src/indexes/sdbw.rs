use super::*;
use std::ops::AddAssign;

pub struct Index {}
impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let features_variances = x.var_axis(Axis(0), 0.);
        let clusters = calc_clusters(&y);
        let cluster_centers = calc_clusters_centers(&clusters, &x);

        let mut stor: Vec<f64> = Vec::with_capacity(clusters.len());
        let mut stor_sqrt: Vec<f64> = Vec::with_capacity(clusters.len());
        for (c, arr) in clusters.iter() {
            let mut row = Array1::zeros(x.ncols());
            for i in arr.iter() {
                row.add_assign(&(&x.row(*i) - &cluster_centers[c]).powi(2));
            }
            row /= arr.len() as f64;
            let vec_length = calc_vector_euclidean_length(&row.view());
            stor.push(vec_length);
            stor_sqrt.push(vec_length.sqrt());
        }
        let S = Array1::from_vec(stor).mean().ok_or("Cant calculate mean")?
            / calc_vector_euclidean_length(&features_variances.view());

        let std = Array1::from_vec(stor_sqrt).mean().ok_or("Cant calc std")?;
        let dencity = |cluster1: i32, cluster2: i32, point: &ArrayView1<f64>| -> usize {
            let mut retval: usize = 0;
            for i in [cluster1, cluster2] {
                for j in clusters[&i].iter() {
                    let dist = find_euclidean_distance(point, &x.row(*j).view());
                    retval += (dist <= std) as u8 as usize;
                }
            }
            retval
        };
        let mut acum: usize = 0;
        for (i, c1) in clusters.keys().enumerate() {
            for (j, c2) in clusters.keys().enumerate() {
                if i < j {
                    let R = dencity(
                        *c1,
                        *c2,
                        &((&cluster_centers[c1] + &cluster_centers[c2]) / 2.0).view(),
                    ) / dencity(*c1, *c2, &cluster_centers[c1].view()).max(dencity(
                        *c1,
                        *c2,
                        &cluster_centers[c2].view(),
                    ));
                    acum += R;
                }
            }
        }

        let q = clusters.keys().len();
        let Dbw = ((1 / (q * (q - 1))) * acum) as f64;

        let value = S + Dbw;
        Ok(value)
    }
}
