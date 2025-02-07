use std::iter::zip;

use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use itertools::Itertools;
use ndarray::{ArcArray1, ArrayView1};

#[derive(Clone, Copy, Debug)]
pub struct CIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        pairs_in_the_same_cluster: &ArrayView1<i8>,
        distances: &ArrayView1<f64>,
    ) -> Result<f64, CalcError> {
        let nw = pairs_in_the_same_cluster
            .iter()
            .filter(|i| **i == 1)
            .count();
        let sorted_distances = distances
            .iter()
            .sorted_unstable_by(|a, b| a.total_cmp(b))
            .map(|e| *e)
            .collect::<Vec<f64>>();

        //calculating sum of Nw minimum and maximum distances
        let mut sum_of_minimum_distances = 0.0;
        let mut sum_of_maximum_distances = 0.0;
        for i in 0..nw {
            sum_of_minimum_distances += sorted_distances[i];
            sum_of_maximum_distances += sorted_distances[(sorted_distances.len() - 1) - i];
        }

        let sw = zip(pairs_in_the_same_cluster, distances)
            .filter(|(p, _)| **p == 1)
            .map(|(_, d)| *d)
            .sum::<f64>();

        Ok((sw - sum_of_minimum_distances) / (sum_of_maximum_distances - sum_of_minimum_distances))
        ////calculating Nw  -- total number of pairs of observations belonging to the same cluster
        //let counts = y
        //    .iter()
        //    .counts()
        //    .iter()
        //    .map(|(i, n)| (**i, (*n * (*n - 1)) / 2))
        //    .collect::<HashMap<i32, usize>>();
        //let number_of_pairs_in_clusters = counts.values().sum::<usize>();
        //
        ////calculating distances beetween all possible pars of points in dataset
        //let mut distances = Vec::with_capacity(y.len() * (y.len() - 1) / 2);
        //let mut distances_per_cluster = counts
        //    .keys()
        //    .map(|i| (*i, 0.0))
        //    .collect::<HashMap<i32, f64>>();
        //
        //for (i, (row1, c1)) in zip(x.rows(), y).enumerate() {
        //    for (j, (row2, c2)) in zip(x.rows(), y).enumerate() {
        //        if i < j {
        //            let dist = (&row2 - &row1).pow2().sum().sqrt();
        //            distances.push(dist);
        //            if c1 == c2 {
        //                distances_per_cluster
        //                    .get_mut(c1)
        //                    .ok_or("cant add distance")?
        //                    .add_assign(dist);
        //            }
        //        }
        //    }
        //}
        //
        ////sorting by min
        //distances.sort_unstable_by(|a, b| a.total_cmp(b));
        //
        ////calculating sum of Nw minimum and maximum distances
        //let mut sum_of_minimum_distances = 0.0;
        //let mut sum_of_maximum_distances = 0.0;
        //for i in 0..number_of_pairs_in_clusters {
        //    sum_of_minimum_distances += distances[i];
        //    sum_of_maximum_distances += distances[(distances.len() - 1) - i];
        //}
        //
        ////calculating Sw -- sum of the within-cluster distances
        //let sum_of_withincluster_distances = distances_per_cluster.values().sum::<f64>();
        //
        ////calculating c_index value
        //let val = (sum_of_withincluster_distances - sum_of_minimum_distances)
        //    / (sum_of_maximum_distances - sum_of_minimum_distances);
        // Ok(val)
    }
}

pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, CIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, CIndexValue>) -> Self {
        Self {
            index: Index,
            sender,
        }
    }
}

impl<'a> Subscriber<(ArcArray1<i8>, ArcArray1<f64>)> for Node<'a> {
    fn recieve_data(&mut self, data: Result<(ArcArray1<i8>, ArcArray1<f64>), CalcError>) {
        let res = match data.as_ref() {
            Ok((p, d)) => self
                .index
                .compute(&p.view(), &d.view())
                .map(|val| CIndexValue { val }),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
