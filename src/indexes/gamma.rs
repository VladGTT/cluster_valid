use core::f64;

use super::*;
use helpers::raw_data::RawDataType;

#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(
        &self,
        distances: &Vec<f64>,
        pairs_in_the_same_cluster: &Vec<i8>,
    ) -> Result<f64, CalcError> {
        let (mut s_plus, mut s_minus) = (0.0, 0.0);

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        //belonging to the same cluster

        for (i, (d1, b1)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(distances, pairs_in_the_same_cluster).enumerate() {
                if i < j && (*b1 == 0 && *b2 == 1) {
                    if d1 < d2 {
                        s_plus += 1.0;
                    }
                    if d1 > d2 {
                        s_minus += 1.0;
                    }
                }
            }
        }
        let value = (s_plus - s_minus) / (s_plus + s_minus);
        Ok(value)
    }
}
#[derive(Default)]
pub struct Node {
    index: Index,
    pub res: Option<Result<f64, CalcError>>,
}

impl<'a> Subscriber<RawDataType<'a>> for Node {
    fn recieve_data(&mut self, data: &RawDataType<'a>) {
        let (x, y) = *data;
        self.res = Some(self.index.compute(x, y));
    }
}
