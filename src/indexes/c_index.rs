use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use itertools::Itertools;
use ndarray::{ArrayView1, ArrayView2};
use std::{collections::HashMap, iter::zip, ops::AddAssign};

#[derive(Clone, Copy, Debug)]
pub struct CIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<CIndexValue, CalcError> {
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
        let val = (sum_of_withincluster_distances - sum_of_minimum_distances)
            / (sum_of_maximum_distances - sum_of_minimum_distances);
        Ok(CIndexValue { val })
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

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for Node<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        let res = match data.as_ref() {
            Ok((x, y)) => self.index.compute(x, y),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
