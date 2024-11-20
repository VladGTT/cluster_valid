use super::*;
use ndarray::Axis;
use ndarray_linalg::Scalar;
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(
        &self,
        x: &ArrayView2<f64>,
        distances: &Vec<f64>,
        pairs_in_the_same_cluster: &Vec<i8>,
    ) -> Result<f64, CalcError> {
        let mut s_minus = 0.0;

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        //belonging to the same cluster

        for (i, (d1, b1)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
                if i < j && (*b1 == 0 && *b2 == 1) && d1 > d2 {
                    s_minus += 1.0;
                }
            }
        }

        let (n, _) = x.dim();
        let n_t = n as f64 * (n as f64 - 1.) / 2.0;
        let value = 2. * s_minus / (n_t * (n_t - 1.0));
        Ok(value)
    }
}
