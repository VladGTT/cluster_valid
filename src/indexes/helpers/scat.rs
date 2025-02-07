use std::iter::zip;

use crate::{
    calc_error::CalcError,
    sender::{Sender, Subscriber},
};
use ndarray::{ArcArray1, Array2, ArrayView1, ArrayView2, Axis};
#[derive(Debug, Clone)]
pub struct ScatValue {
    pub val: f64,
    pub clusters_vars: ArcArray1<f64>,
    pub var: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<(f64, ArcArray1<f64>, f64), CalcError> {
        let var = x.var_axis(Axis(0), 0.);
        let q = *y.iter().max().ok_or("Cant get max cluster index")? as usize + 1;
        let mut clusters_vars: Array2<f64> = Array2::zeros((q, x.ncols()));
        for i in 0..q {
            let vec = zip(x.rows(), y)
                .filter(|(_, c)| **c as usize == i)
                .map(|(v, _)| v.into_iter())
                .flatten()
                .map(|v| *v)
                .collect::<Vec<f64>>();
            let arr =
                Array2::from_shape_vec((vec.len(), x.ncols()), vec).map_err(|e| e.to_string())?;
            let var = arr.var_axis(Axis(0), 0.);
            clusters_vars.row_mut(i).assign(&var);
        }

        let var = var.pow2().sum().sqrt();
        let clusters_vars = clusters_vars.dot(&clusters_vars.t()).diag().sqrt();
        let clusters_vars_mean = clusters_vars.mean().ok_or("Cant calc mean")?;
        let val = clusters_vars_mean / var;
        Ok((val, clusters_vars.to_shared(), var))
    }
}
pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, ScatValue>,
}
impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, ScatValue>) -> Self {
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
        let res = match data {
            Ok((ref x, ref y)) => {
                self.index
                    .compute(x, y)
                    .map(|(val, clusters_vars, var)| ScatValue {
                        val,
                        clusters_vars,
                        var,
                    })
            }
            Err(err) => Err(err),
        };
        self.sender.send_to_subscribers(res);
    }
}
