use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{ArcArray1, ArcArray2, Array1, ArrayView1, ArrayView2, Axis};
use std::{collections::HashMap, sync::Arc};

use crate::sender::{Sender, Subscriber};

use super::helpers::{between_group_dispercion::BGDValue, counts, total_dispercion::TDValue};

#[derive(Clone, Copy, Debug)]
pub struct RatkowskyIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(
        &self,
        counts: &ArrayView1<usize>,
        tg: &ArrayView2<f64>,
        bg: &ArrayView2<f64>,
    ) -> Result<f64, CalcError> {
        let diag_bg = bg.diag();
        let diag_tg = tg.diag();
        let div = &diag_bg / &diag_tg;
        let r = div.mean().ok_or(CalcError::from("Cant calculate mean"))?;
        let q = counts.len() as f64;
        let value = (r / q).sqrt();
        Ok(value)
        // let x_mean = x
        //     .mean_axis(Axis(0))
        //     .ok_or("Cant compute mean for dataset")?;
        //
        // let (num_of_elems, num_of_vars) = x.dim();
        //
        // let mut bgss: Array1<f64> = Array1::zeros(num_of_vars);
        // for (i, c) in clusters_centroids {
        //     bgss = bgss + clusters[&i].len() as f64 * (c - &x_mean).pow2();
        // }
        //
        // let tss = x.var_axis(Axis(0), 0.) * num_of_elems as f64;
        //
        // let s_squared = (bgss / tss).sum() / num_of_vars as f64;
        // let value = (s_squared / clusters.keys().len() as f64).sqrt();
        // Ok(value)
    }
}

pub struct Node<'a> {
    index: Index,
    bg: Option<Result<ArcArray2<f64>, CalcError>>,
    tg: Option<Result<ArcArray2<f64>, CalcError>>,
    counts: Option<Result<ArcArray1<usize>, CalcError>>,
    sender: Sender<'a, RatkowskyIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(counts), Some(tg), Some(bg)) =
            (self.counts.as_ref(), self.tg.as_ref(), self.bg.as_ref())
        {
            let res = match counts.combine(tg).combine(bg) {
                Ok(((cnts, tg), bg)) => self
                    .index
                    .compute(&cnts.view(), &tg.view(), &bg.view())
                    .map(|val| RatkowskyIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.counts = None;
            self.tg = None;
            self.bg = None;
        }
    }
    pub fn new(sender: Sender<'a, RatkowskyIndexValue>) -> Self {
        Self {
            index: Index,
            counts: None,
            tg: None,
            bg: None,
            sender,
        }
    }
}

impl<'a> Subscriber<ArcArray1<usize>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<ArcArray1<usize>, CalcError>) {
        self.counts = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<TDValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<TDValue, CalcError>) {
        self.tg = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}
impl<'a> Subscriber<BGDValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<BGDValue, CalcError>) {
        self.bg = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}
