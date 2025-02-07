use crate::{
    calc_error::{CalcError, CombineErrors},
    indexes::helpers::clusters_centroids::ClustersCentroidsValue,
    sender::{Sender, Subscriber},
};

use ndarray::{ArcArray2, Array2, ArrayView1, ArrayView2};
use std::iter::zip;

#[derive(Clone, Debug)]
pub struct WGDValue {
    pub val: ArcArray2<f64>,
}
#[derive(Default)]
pub struct WGD;
impl WGD {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        clusters_centroids: &ArrayView2<f64>,
    ) -> Result<ArcArray2<f64>, CalcError> {
        let (n, d) = x.dim();
        let mut dif: Array2<f64> = Array2::zeros((n, d));
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            let temp = &clusters_centroids.row(*y as usize) - &x;
            dif.row_mut(i).assign(&temp);
        }
        let wg = dif.t().dot(&dif);
        Ok(wg.into_shared())
    }
}
pub struct WGDNode<'a> {
    index: WGD,
    clusters_centroids: Option<Result<ArcArray2<f64>, CalcError>>,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    sender: Sender<'a, WGDValue>,
}
impl<'a> WGDNode<'a> {
    pub fn new(sender: Sender<'a, WGDValue>) -> Self {
        Self {
            index: WGD,
            clusters_centroids: None,
            raw_data: None,
            sender,
        }
    }
    fn process_when_ready(&mut self) {
        if let (Some(clusters_centroids), Some(raw_data)) =
            (self.clusters_centroids.as_ref(), self.raw_data.as_ref())
        {
            let res = match clusters_centroids.combine(raw_data) {
                Ok((cls_ctrs, (x, y))) => self
                    .index
                    .compute(x, y, &cls_ctrs.view())
                    .map(|val| WGDValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.clusters_centroids = None;
        }
    }
}
impl<'a> Subscriber<ClustersCentroidsValue> for WGDNode<'a> {
    fn recieve_data(&mut self, data: Result<ClustersCentroidsValue, CalcError>) {
        self.clusters_centroids = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for WGDNode<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        self.raw_data = Some(data);
        self.process_when_ready();
    }
}
