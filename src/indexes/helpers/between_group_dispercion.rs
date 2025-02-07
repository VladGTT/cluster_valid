use crate::{
    calc_error::{CalcError, CombineErrors},
    sender::{Sender, Subscriber},
};
use ndarray::{ArcArray2, Array2, ArrayView1, ArrayView2, Axis};

use super::clusters_centroids::ClustersCentroidsValue;

#[derive(Clone, Debug)]
pub struct BGDValue {
    pub val: ArcArray2<f64>,
}

#[derive(Default)]
pub struct BGD;
impl BGD {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        clusters_centroids: &ArrayView2<f64>,
    ) -> Result<ArcArray2<f64>, CalcError> {
        let (n, d) = x.dim();
        let data_center = x.mean_axis(Axis(0)).ok_or("Cant calc data centroid")?;
        let mut b: Array2<f64> = Array2::zeros((n, d));
        for (i, y) in y.iter().enumerate() {
            let temp = &data_center - &clusters_centroids.row(*y as usize);
            b.row_mut(i).assign(&temp);
        }
        let bg = b.t().dot(&b);
        Ok(bg.into_shared())
    }
}
pub struct BGDNode<'a> {
    index: BGD,
    clusters_centroids: Option<Result<ArcArray2<f64>, CalcError>>,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    sender: Sender<'a, BGDValue>,
}
impl<'a> BGDNode<'a> {
    pub fn new(sender: Sender<'a, BGDValue>) -> Self {
        Self {
            index: BGD,
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
                    .map(|val| BGDValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.clusters_centroids = None;
        }
    }
}
impl<'a> Subscriber<ClustersCentroidsValue> for BGDNode<'a> {
    fn recieve_data(&mut self, data: Result<ClustersCentroidsValue, CalcError>) {
        self.clusters_centroids = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for BGDNode<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        self.raw_data = Some(data);
        self.process_when_ready();
    }
}
