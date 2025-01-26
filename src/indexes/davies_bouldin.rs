use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::{collections::HashMap, sync::Arc};

use crate::sender::{Sender, Subscriber};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct DaviesBouldinIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        clusters_centroids: &HashMap<i32, Array1<f64>>,
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> Result<f64, CalcError> {
        let mut stor: HashMap<i32, f64> = HashMap::default();

        for (c, arr) in clusters.iter() {
            let temp = arr
                .par_iter()
                .map(|i| (&x.row(*i) - &clusters_centroids[c]).pow2().sum().sqrt())
                .sum::<f64>()
                / arr.len() as f64;
            stor.insert(*c, temp);
        }
        let q = clusters.keys().len();

        let mut acum = 0.0;
        let mut temp: Vec<f64> = Vec::with_capacity(q);
        for i in clusters.keys() {
            for j in clusters.keys() {
                if *i != *j {
                    let coef = (stor[i] + stor[j])
                        / (&clusters_centroids[j] - &clusters_centroids[i])
                            .pow2()
                            .sum()
                            .sqrt();
                    temp.push(coef);
                }
            }
            acum += temp
                .iter()
                .max_by(|a, b| a.total_cmp(b))
                .ok_or("Cant find max val")?;
            temp.clear();
        }

        let val = acum / q as f64;
        Ok(val)
    }
}

pub struct Node<'a> {
    index: Index,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    clusters: Option<Result<Arc<HashMap<i32, Array1<usize>>>, CalcError>>,
    clusters_centroids: Option<Result<Arc<HashMap<i32, Array1<f64>>>, CalcError>>,
    sender: Sender<'a, DaviesBouldinIndexValue>,
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
                    .map(|val| DaviesBouldinIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.raw_data = None;
            self.clusters = None;
            self.clusters_centroids = None;
        }
    }
    pub fn new(sender: Sender<'a, DaviesBouldinIndexValue>) -> Self {
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
