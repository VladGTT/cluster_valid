use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use std::{collections::HashMap, sync::Arc};

use super::{Sender, Subscriber};
use rayon::prelude::*;
pub struct Index {}
impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        clusters_centroids: &HashMap<i32, Array1<f64>>,
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> Result<f64, CalcError> {
        if y.iter().counts().keys().len() != 2 {
            return Err(CalcError::from("There is more than 2 clusters"));
        }

        let dataset_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;

        let within_group_dispersion_parent = {
            let diff = &x - &dataset_mean;
            diff.dot(&diff.t()).diag().sum()
        };

        let mut within_group_dispersion_children: HashMap<i32, f64> = HashMap::default();
        for (cl_idx, idxs) in clusters.iter() {
            for idx in idxs {
                let diff = &x.row(*idx) - &cluster_centers[cl_idx];

                within_group_dispersion_children
                    .entry(*cl_idx)
                    .and_modify(|v| *v += diff.dot(&diff))
                    .or_insert(0.);
            }
        }

        let value = within_group_dispersion_parent
            / (within_group_dispersion_children.values().sum::<f64>());
        Ok(value)
    }
}
