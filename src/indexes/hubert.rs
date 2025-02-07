use crate::calc_error::{CalcError, CombineErrors};
use crate::sender::{Sender, Subscriber};
use ndarray::{ArcArray2, Array2, ArrayView1, ArrayView2};

use super::helpers::clusters_centroids::ClustersCentroidsValue;
use std::iter::zip;
#[derive(Clone, Copy, Debug)]
pub struct HubertIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        clusters_centroids: &ArrayView2<f64>,
    ) -> Result<f64, CalcError> {
        let mut P: Array2<f64> = Array2::zeros((x.nrows(), x.nrows()));
        let mut Q: Array2<f64> = Array2::zeros(P.dim());
        for (i, (row1, c1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, c2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = (&row2 - &row1).pow2().sum().sqrt();
                    *P.get_mut((i, j)).ok_or("Cant get elem")? = dist;
                    *P.get_mut((j, i)).ok_or("Cant get elem")? = dist;
                    let centroids_dist = (&clusters_centroids.row(*c2 as usize)
                        - &clusters_centroids.row(*c1 as usize))
                        .pow2()
                        .sum()
                        .sqrt();
                    *Q.get_mut((i, j)).ok_or("Cant get elem")? = centroids_dist;
                    *Q.get_mut((j, i)).ok_or("Cant get elem")? = centroids_dist;
                }
            }
        }
        let mean_P = P.mean().ok_or("Cant calc P mean")?;
        let mean_Q = Q.mean().ok_or("Cant calc Q mean")?;

        let var_P = P.var(0.);
        let var_Q = Q.var(0.);

        let centered_P = &P - mean_P;
        let centered_Q = &Q - mean_Q;

        let flattened_P = centered_P.flatten();
        let flattened_Q = centered_Q.flatten();

        let temp = flattened_P.dot(&flattened_Q);
        let val = temp / (var_P * var_Q);

        Ok(val)
    }
}
pub struct Node<'a> {
    index: Index,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    clusters_centroids: Option<Result<ArcArray2<f64>, CalcError>>,
    sender: Sender<'a, HubertIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(clusters_centroids)) =
            (self.raw_data.as_ref(), self.clusters_centroids.as_ref())
        {
            let res = match raw_data.combine(clusters_centroids) {
                Ok(((x, y), cls_ctrds)) => self
                    .index
                    .compute(x, y, &cls_ctrds.view())
                    .map(|val| HubertIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.clusters_centroids = None;
        }
    }
    pub fn new(sender: Sender<'a, HubertIndexValue>) -> Self {
        Self {
            index: Index,
            raw_data: None,
            clusters_centroids: None,
            sender,
        }
    }
}

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for Node<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        self.raw_data = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<ClustersCentroidsValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<ClustersCentroidsValue, CalcError>) {
        self.clusters_centroids = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}
