use super::*;

pub struct Index {}

impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        let mut distances: Vec<f64> = Vec::default();
        let mut pairs_in_the_same_cluster: Vec<i8> = Vec::default();

        //calculating distances beetween pair of points and does they belong to the same cluster
        for (i, (row1, cluster1)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
            for (j, (row2, cluster2)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
                if i < j {
                    pairs_in_the_same_cluster.push((cluster1 != cluster2) as i8); // the same cluster = 0, different = 1
                    distances.push(find_euclidean_distance(&row1, &row2));
                }
            }
        }

        let total_number_of_pairs = x.len() * (x.len() - 1) / 2;
        let (mut s_plus, mut s_minus): (usize, usize) = (0, 0);

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        // belonging to the same cluster
        let mut t: usize = 0;
        for (i, (d1, b1)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
                if i < j {
                    if (*b1 == 1 && *b2 == 1) || (*b1 == 0 && *b2 == 0) {
                        t += 1;
                    }
                    if *b1 == 0 && *b2 == 1 {
                        s_plus += (d1 < d2) as u8 as usize;
                        s_minus += (d1 > d2) as u8 as usize;
                    }
                }
            }
        }
        let v0 = (total_number_of_pairs * (total_number_of_pairs - 1)) as f64 / 2.0;
        let value = (s_plus - s_minus) as f64 / ((v0 - t as f64) * v0).sqrt();
        Ok(value)
    }
}
