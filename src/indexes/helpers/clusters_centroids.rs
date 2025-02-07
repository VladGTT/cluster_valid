use crate::{
    calc_error::{CalcError, CombineErrors},
    sender::{Sender, Subscriber},
};
use ndarray::{ArcArray1, ArcArray2, Array2, ArrayView1, ArrayView2};
use std::iter::zip;
#[derive(Clone, Debug)]
pub struct ClustersCentroidsValue {
    pub val: ArcArray2<f64>,
}
#[derive(Default)]
pub struct ClustersCentroids;
impl ClustersCentroids {
    pub fn compute(
        &self,
        data: &ArrayView2<f64>,
        clusters: &ArrayView1<i32>,
        counts: &ArrayView1<usize>,
    ) -> Result<ArcArray2<f64>, CalcError> {
        let q = counts.len();
        let mut centroids: Array2<f64> = Array2::default((q, data.ncols()));
        for (x, y) in zip(data.rows(), clusters.iter()) {
            let mut r = centroids.row_mut(*y as usize);
            r += &(&x / counts[*y as usize] as f64);
        }
        let res = centroids.into_shared();
        Ok(res)
    }
}
pub struct ClustersCentroidsNode<'a> {
    index: ClustersCentroids,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    counts: Option<Result<ArcArray1<usize>, CalcError>>,
    sender: Sender<'a, ClustersCentroidsValue>,
}
impl<'a> ClustersCentroidsNode<'a> {
    pub fn new(sender: Sender<'a, ClustersCentroidsValue>) -> Self {
        Self {
            index: ClustersCentroids,
            raw_data: None,
            counts: None,
            sender,
        }
    }
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(counts)) = (self.raw_data.as_ref(), self.counts.as_ref()) {
            let res = match raw_data.combine(counts) {
                Ok(((ref x, ref y), cnts)) => self
                    .index
                    .compute(x, y, &cnts.view())
                    .map(|val| ClustersCentroidsValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.counts = None;
        }
    }
}

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for ClustersCentroidsNode<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        self.raw_data = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<ArcArray1<usize>> for ClustersCentroidsNode<'a> {
    fn recieve_data(&mut self, data: Result<ArcArray1<usize>, CalcError>) {
        self.counts = Some(data);
        self.process_when_ready();
    }
}
