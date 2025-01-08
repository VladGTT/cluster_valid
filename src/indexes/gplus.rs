use super::*;
use helpers::{pairs_and_distances::PairsAndDistancesType, raw_data::RawDataType};
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
#[derive(Default)]
pub struct Node<'a> {
    index: Index,

    raw_data: Option<RawDataType<'a>>,
    pairs_and_distances: Option<PairsAndDistancesType>,
    pub res: Option<Result<f64, CalcError>>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some((x, _)), Some((pairs, distances))) =
            (self.raw_data.as_ref(), self.pairs_and_distances.as_ref())
        {
            self.res = Some(self.index.compute(x, distances, pairs));
        }
    }
}

impl<'a> Subscriber<PairsAndDistancesType> for Node<'a> {
    fn recieve_data(&mut self, data: &PairsAndDistancesType) {
        self.pairs_and_distances = Some(data.clone());
        self.process_when_ready();
    }
}
impl<'a> Subscriber<RawDataType<'a>> for Node<'a> {
    fn recieve_data(&mut self, data: &RawDataType<'a>) {
        self.raw_data = Some(*data);
        self.process_when_ready();
    }
}
