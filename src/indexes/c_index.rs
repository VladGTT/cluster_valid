use super::*;
use calc_error::CalcError;
use helpers::raw_data::RawDataType;
use itertools::Itertools;
use std::iter::zip;
use std::ops::AddAssign;

#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(&self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<f64, CalcError> {
        //calculating Nw  -- total number of pairs of observations belonging to the same cluster
        let counts = y
            .iter()
            .counts()
            .iter()
            .map(|(i, n)| (**i, (*n * (*n - 1)) / 2))
            .collect::<HashMap<i32, usize>>();
        let number_of_pairs_in_clusters = counts.values().sum::<usize>();

        //calculating distances beetween all possible pars of points in dataset
        let mut distances = Vec::with_capacity(y.len() * (y.len() - 1) / 2);
        let mut distances_per_cluster = counts
            .keys()
            .map(|i| (*i, 0.0))
            .collect::<HashMap<i32, f64>>();

        for (i, (row1, c1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, c2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = (&row2 - &row1).pow2().sum().sqrt();
                    distances.push(dist);
                    if c1 == c2 {
                        distances_per_cluster
                            .get_mut(c1)
                            .ok_or("cant add distance")?
                            .add_assign(dist);
                    }
                }
            }
        }

        //sorting by min
        distances.sort_unstable_by(|a, b| a.total_cmp(b));

        //calculating sum of Nw minimum and maximum distances
        let mut sum_of_minimum_distances = 0.0;
        let mut sum_of_maximum_distances = 0.0;
        for i in 0..number_of_pairs_in_clusters {
            sum_of_minimum_distances += distances[i];
            sum_of_maximum_distances += distances[(distances.len() - 1) - i];
        }

        //calculating Sw -- sum of the within-cluster distances
        let sum_of_withincluster_distances = distances_per_cluster.values().sum::<f64>();

        //calculating c_index value
        let value = (sum_of_withincluster_distances - sum_of_minimum_distances)
            / (sum_of_maximum_distances - sum_of_minimum_distances);
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
