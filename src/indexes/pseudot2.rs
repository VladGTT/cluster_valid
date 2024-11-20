use super::*;
use itertools::Itertools;

pub struct Index {}
impl Computable for Index {
    fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
        if y.iter().counts().keys().len() != 2 {
            return Err(CalcError::from("There is more than 2 clusters"));
        }
        let dataset_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
        let clusters = calc_clusters(&y);
        let cluster_centers = calc_clusters_centers(&clusters, &x);

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
        let w_children_sum = within_group_dispersion_children.values().sum::<f64>();
        let value = (within_group_dispersion_parent - w_children_sum)
            / w_children_sum
            / (x.nrows() - 2) as f64;
        Ok(value)
    }
}
