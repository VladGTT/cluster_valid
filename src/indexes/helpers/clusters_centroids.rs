use crate::{
    calc_error::{CalcError, CombineErrors},
    sender::{Sender, Subscriber},
};
use ndarray::{prelude::*, Array1, ArrayView2};
use rayon::prelude::*;
use std::{collections::HashMap, sync::Arc};
#[derive(Default)]
pub struct ClustersCentroids;
impl ClustersCentroids {
    pub fn compute(
        &self,
        clusters_indexes: &HashMap<i32, Array1<usize>>,
        data: &ArrayView2<f64>,
    ) -> Result<HashMap<i32, Array1<f64>>, CalcError> {
        let res = clusters_indexes
            .into_par_iter()
            .map(|(c, arr)| {
                let mut sum: Array1<f64> = Array1::zeros(data.ncols());
                for i in arr {
                    sum += &data.row(*i);
                }
                (*c, sum / arr.len() as f64)
            })
            .collect::<HashMap<i32, Array1<f64>>>();
        Ok(res)
    }
}
pub struct ClustersCentroidsNode<'a> {
    index: ClustersCentroids,
    clusters: Option<Result<Arc<HashMap<i32, Array1<usize>>>, CalcError>>,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    sender: Sender<'a, Arc<HashMap<i32, Array1<f64>>>>,
}
impl<'a> ClustersCentroidsNode<'a> {
    pub fn new(sender: Sender<'a, Arc<HashMap<i32, Array1<f64>>>>) -> Self {
        Self {
            index: ClustersCentroids,
            clusters: None,
            raw_data: None,
            sender,
        }
    }
    fn process_when_ready(&mut self) {
        if let (Some(clusters), Some(raw_data)) = (self.clusters.as_ref(), self.raw_data.as_ref()) {
            let res = match clusters.combine(raw_data) {
                Ok((clstrs, (x, _))) => self.index.compute(clstrs, x).map(|v| Arc::new(v)),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.clusters = None;
        }
    }
}
impl<'a> Subscriber<Arc<HashMap<i32, Array1<usize>>>> for ClustersCentroidsNode<'a> {
    fn recieve_data(&mut self, data: Result<Arc<HashMap<i32, Array1<usize>>>, CalcError>) {
        self.clusters = Some(data);
        self.process_when_ready();
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
