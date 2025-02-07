use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::iter::zip;
#[derive(Clone, Copy, Debug)]
pub struct SilhouetteIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(&self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<f64, CalcError> {
        let q = *y.iter().max().ok_or("Cant get numb of clusters")? as usize + 1;
        let mut s: Vec<Vec<f64>> = Vec::new();
        s.resize(q, Vec::default());
        for (row1, c1) in zip(x.rows(), y) {
            let mut d: Vec<Vec<f64>> = Vec::new();
            d.resize(q, Vec::default());
            for (row2, c2) in zip(x.rows(), y) {
                let dist = (&row2 - &row1).pow2().sum().sqrt();
                if row1 != row2 {
                    d.get_mut(*c2 as usize).ok_or("Cant get val")?.push(dist);
                }
            }
            let mut d = d
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    if i == *c1 as usize {
                        Array1::from_vec(v).mean()
                    } else {
                        Some(v.iter().sum::<f64>() / (v.len() - 1) as f64)
                    }
                })
                .collect::<Option<Vec<f64>>>()
                .ok_or("Cant calc mean")?;
            let a = d.remove(*c1 as usize);
            let b = d
                .into_iter()
                .min_by(|a, b| a.total_cmp(b))
                .ok_or("Cant calc min")?;
            s.get_mut(*c1 as usize)
                .ok_or("Cant get val")?
                .push((b - a) / b.max(a));
        }
        let s = s
            .into_iter()
            .map(|v| Array1::from_vec(v).mean())
            .collect::<Option<Vec<f64>>>()
            .ok_or("Cant calc mean")?;
        Array1::from_vec(s)
            .mean()
            .ok_or("Cant calc res mean".into())

        // let mut temp: Vec<f64> = Vec::with_capacity(clusters.keys().len() - 1);
        // let mut stor: Vec<f64> = Vec::with_capacity(x.nrows());
        // for (c, arr) in clusters.iter() {
        //     for i in arr {
        //         let mut sum_inter_dists = 0.0;
        //         let row = x.row(*i);
        //         for j in arr {
        //             if i != j {
        //                 sum_inter_dists += (&x.row(*j) - &row).pow2().sum().sqrt();
        //             }
        //         }
        //         for (c2, arr2) in clusters.iter() {
        //             if c2 != c {
        //                 let mut sum_intra_dists = 0.0;
        //                 for j2 in arr2 {
        //                     sum_intra_dists += (&x.row(*j2) - &row).pow2().sum().sqrt();
        //                 }
        //                 temp.push(sum_intra_dists / arr2.len() as f64);
        //             }
        //         }
        //         let a = sum_inter_dists
        //             / if arr.len() == 1 {
        //                 1.0
        //             } else {
        //                 (arr.len() - 1) as f64
        //             };
        //         let b = temp
        //             .iter()
        //             .min_by(|a, b| a.total_cmp(b))
        //             .ok_or("Cant find min")?;
        //
        //         stor.push((b - a) / a.max(*b));
        //         temp.clear()
        //     }
        // }
        // let val = Array1::from_vec(stor).mean().ok_or("Cant calc mean")?;
        // Ok(val)
    }
}

pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, SilhouetteIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, SilhouetteIndexValue>) -> Self {
        Self {
            index: Index::default(),
            sender,
        }
    }
}
impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for Node<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        let res = match data.as_ref() {
            Ok((x, y)) => self
                .index
                .compute(x, y)
                .map(|val| SilhouetteIndexValue { val }),

            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
