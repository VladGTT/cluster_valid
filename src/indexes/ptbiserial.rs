use super::*;

pub struct Index {}
impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let (mut num_pairs_the_same_clust, mut num_pairs_dif_clust): (usize, usize) = (0, 0);
        let (mut sum_dist_same_clust, mut sum_dist_dif_clust): (f64, f64) = (0.0, 0.0);
        let num_pairs_total = x.nrows() * (x.nrows() - 1) / 2;
        let mut distances: Array1<f64> = Array1::zeros(num_pairs_total);
        let mut ctr = 0;
        for (i, (row1, clust1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, clust2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = find_euclidean_distance(&row1, &row2);
                    distances[ctr] = dist;
                    if clust1 == clust2 {
                        sum_dist_same_clust += dist;
                        num_pairs_the_same_clust += 1;
                    } else {
                        sum_dist_dif_clust += dist;
                        num_pairs_dif_clust += 1;
                    }
                    ctr += 1;
                }
            }
        }
        let std = distances.std(0.);

        let (num_pairs_the_same_clust, num_pairs_dif_clust, num_pairs_total) = (
            num_pairs_the_same_clust as f64,
            num_pairs_dif_clust as f64,
            num_pairs_total as f64,
        );

        let value = (sum_dist_same_clust / num_pairs_the_same_clust
            - sum_dist_dif_clust / num_pairs_dif_clust)
            * (num_pairs_dif_clust * num_pairs_the_same_clust).sqrt()
            / num_pairs_total;

        Ok(value)
    }
}
