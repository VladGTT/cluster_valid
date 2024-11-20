use super::*;

pub struct Index {}
impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let (mut num_pairs_the_same_clust, mut num_pairs_dif_clust): (f64, f64) = (0., 0.);
        let (mut sum_dist_same_clust, mut sum_dist_dif_clust): (f64, f64) = (0.0, 0.0);
        for (i, (row1, clust1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, clust2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = find_euclidean_distance(&row1, &row2);
                    if clust1 == clust2 {
                        sum_dist_same_clust += dist;
                        num_pairs_the_same_clust += 1.;
                    } else {
                        sum_dist_dif_clust += dist;
                        num_pairs_dif_clust += 1.;
                    }
                }
            }
        }
        let value = (sum_dist_same_clust / num_pairs_the_same_clust)
            / (sum_dist_dif_clust / num_pairs_dif_clust);
        Ok(value)
    }
}
