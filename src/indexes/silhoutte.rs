use super::{Sender, Subscriber};
use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::{collections::HashMap, sync::Arc};
#[derive(Clone, Copy, Debug)]
pub struct SilhoutteIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> Result<SilhoutteIndexValue, CalcError> {
        let mut temp: Vec<f64> = Vec::with_capacity(clusters.keys().len() - 1);
        let mut stor: Vec<f64> = Vec::with_capacity(x.nrows());
        for (c, arr) in clusters.iter() {
            for i in arr {
                let mut sum_inter_dists = 0.0;
                let row = x.row(*i);
                for j in arr {
                    if i != j {
                        sum_inter_dists += (&x.row(*j) - &row).pow2().sum().sqrt();
                    }
                }
                for (c2, arr2) in clusters.iter() {
                    if c2 != c {
                        let mut sum_intra_dists = 0.0;
                        for j2 in arr2 {
                            sum_intra_dists += (&x.row(*j2) - &row).pow2().sum().sqrt();
                        }
                        temp.push(sum_intra_dists / arr2.len() as f64);
                    }
                }
                let a = sum_inter_dists
                    / if arr.len() == 1 {
                        1.0
                    } else {
                        (arr.len() - 1) as f64
                    };
                let b = temp
                    .iter()
                    .min_by(|a, b| a.total_cmp(b))
                    .ok_or("Cant find min")?;

                stor.push((b - a) / a.max(*b));
                temp.clear()
            }
        }
        let val = Array1::from_vec(stor).mean().ok_or("Cant calc mean")?;
        Ok(SilhoutteIndexValue { val })
    }
}

pub struct Node<'a> {
    index: Index,
    raw_data: Option<Arc<Result<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>), CalcError>>>,
    clusters: Option<Arc<Result<HashMap<i32, Array1<usize>>, CalcError>>>,
    sender: Sender<'a, SilhoutteIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(clusters)) = (self.raw_data.as_ref(), self.clusters.as_ref()) {
            let res = match raw_data.combine(clusters) {
                Ok(((x, _), cls)) => self.index.compute(x, cls),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(Arc::new(res));
            self.raw_data = None;
            self.clusters = None;
        }
    }
    pub fn new(sender: Sender<'a, SilhoutteIndexValue>) -> Self {
        Self {
            index: Index::default(),
            raw_data: None,
            clusters: None,
            sender,
        }
    }
}
impl<'a> Subscriber<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)> for Node<'a> {
    fn recieve_data(
        &mut self,
        data: Arc<Result<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>), CalcError>>,
    ) {
        self.raw_data = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<HashMap<i32, Array1<usize>>> for Node<'a> {
    fn recieve_data(&mut self, data: Arc<Result<HashMap<i32, Array1<usize>>, CalcError>>) {
        self.clusters = Some(data);
        self.process_when_ready();
    }
}
