use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use std::{collections::HashMap, sync::Arc};

use crate::sender::{Sender, Subscriber};

#[derive(Clone, Copy, Debug)]
pub struct RatkowskyIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(
        &self,
        x: &ArrayView2<f64>,
        clusters_centroids: &HashMap<i32, Array1<f64>>,
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> Result<f64, CalcError> {
        let x_mean = x
            .mean_axis(Axis(0))
            .ok_or("Cant compute mean for dataset")?;

        let (num_of_elems, num_of_vars) = x.dim();

        let mut bgss: Array1<f64> = Array1::zeros(num_of_vars);
        for (i, c) in clusters_centroids {
            bgss = bgss + clusters[&i].len() as f64 * (c - &x_mean).pow2();
        }

        let tss = x.var_axis(Axis(0), 0.) * num_of_elems as f64;

        let s_squared = (bgss / tss).sum() / num_of_vars as f64;
        let value = (s_squared / clusters.keys().len() as f64).sqrt();
        Ok(value)
    }
}

pub struct Node<'a> {
    index: Index,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    clusters: Option<Result<Arc<HashMap<i32, Array1<usize>>>, CalcError>>,
    clusters_centroids: Option<Result<Arc<HashMap<i32, Array1<f64>>>, CalcError>>,
    sender: Sender<'a, RatkowskyIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(clusters), Some(clusters_centroids)) = (
            self.raw_data.as_ref(),
            self.clusters.as_ref(),
            self.clusters_centroids.as_ref(),
        ) {
            let res = match raw_data.combine(clusters).combine(clusters_centroids) {
                Ok((((x, _), cls), cls_ctrds)) => self
                    .index
                    .compute(x, cls_ctrds, cls)
                    .map(|val| RatkowskyIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.clusters = None;
            self.clusters_centroids = None;
        }
    }
    pub fn new(sender: Sender<'a, RatkowskyIndexValue>) -> Self {
        Self {
            index: Index,
            raw_data: None,
            clusters_centroids: None,
            clusters: None,
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
impl<'a> Subscriber<Arc<HashMap<i32, Array1<usize>>>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<Arc<HashMap<i32, Array1<usize>>>, CalcError>) {
        self.clusters = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<Arc<HashMap<i32, Array1<f64>>>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<Arc<HashMap<i32, Array1<f64>>>, CalcError>) {
        self.clusters_centroids = Some(data);
        self.process_when_ready();
    }
}
