use std::iter::zip;

use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{ArcArray1, ArcArray2, ArrayView1, ArrayView2};

use crate::sender::{Sender, Subscriber};

use super::helpers::{clusters_centroids::ClustersCentroidsValue, scat::ScatValue};

#[derive(Clone, Copy, Debug)]
pub struct SDBWIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        scat: &f64,
        centroid_vars: &ArrayView1<f64>,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        clusters_centroids: &ArrayView2<f64>,
    ) -> Result<f64, CalcError> {
        let q = clusters_centroids.nrows();
        let stdev = centroid_vars.sum().sqrt() / q as f64;
        let mut accum = 0;
        for i in 0..q {
            for j in 0..q {
                if i != j {
                    let density1 = Self::density(
                        stdev,
                        x,
                        y,
                        |v| v == i || v == j,
                        ((&clusters_centroids.row(i) + &clusters_centroids.row(j)) / 2.).view(),
                    );
                    let density2 =
                        Self::density(stdev, x, y, |v| v == i, clusters_centroids.row(i));
                    let density3 =
                        Self::density(stdev, x, y, |v| v == j, clusters_centroids.row(j));
                    accum += density1 / density2.max(density3);
                }
            }
        }
        let value = (accum / (q * (q - 1))) as f64;
        Ok(value)
    }
    fn density<F>(
        stdev: f64,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        predicat: F,
        center: ArrayView1<f64>,
    ) -> usize
    where
        F: Fn(usize) -> bool,
    {
        let mut retval: usize = 0;
        for (row, c) in zip(x.rows(), y) {
            if predicat(*c as usize) {
                let dist = (&row - &center).pow2().sum().sqrt();
                retval += (dist <= stdev) as usize;
            }
        }
        retval
    }
}

pub struct Node<'a> {
    index: Index,
    raw_data: Option<Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>>,
    scat: Option<Result<(f64, ArcArray1<f64>, f64), CalcError>>,
    clusters_centroids: Option<Result<ArcArray2<f64>, CalcError>>,
    sender: Sender<'a, SDBWIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(scat), Some(clusters_centroids), Some(raw_data)) = (
            self.scat.as_ref(),
            self.clusters_centroids.as_ref(),
            self.raw_data.as_ref(),
        ) {
            let res = match scat.combine(clusters_centroids).combine(raw_data) {
                Ok((((val, centroid_vars, _), cls_ctrds), (x, y))) => self
                    .index
                    .compute(val, &centroid_vars.view(), x, y, &cls_ctrds.view())
                    .map(|val| SDBWIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.scat = None;
            self.clusters_centroids = None;
        }
    }
    pub fn new(sender: Sender<'a, SDBWIndexValue>) -> Self {
        Self {
            index: Index,
            raw_data: None,
            scat: None,
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
impl<'a> Subscriber<ScatValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<ScatValue, CalcError>) {
        self.scat = Some(data.map(|v| (v.val, v.clusters_vars, v.var)));
        self.process_when_ready();
    }
}
impl<'a> Subscriber<ClustersCentroidsValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<ClustersCentroidsValue, CalcError>) {
        self.clusters_centroids = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}
// use super::*;
// use std::ops::AddAssign;
//
// pub struct Index {}
// impl Computable for Index {
//     fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
//         let features_variances = x.var_axis(Axis(0), 0.);
//         let clusters = calc_clusters(&y);
//         let cluster_centers = calc_clusters_centers(&clusters, &x);
//
//         let mut stor: Vec<f64> = Vec::with_capacity(clusters.len());
//         let mut stor_sqrt: Vec<f64> = Vec::with_capacity(clusters.len());
//         for (c, arr) in clusters.iter() {
//             let mut row = Array1::zeros(x.ncols());
//             for i in arr.iter() {
//                 row.add_assign(&(&x.row(*i) - &cluster_centers[c]).powi(2));
//             }
//             row /= arr.len() as f64;
//             let vec_length = calc_vector_euclidean_length(&row.view());
//             stor.push(vec_length);
//             stor_sqrt.push(vec_length.sqrt());
//         }
//         let S = Array1::from_vec(stor).mean().ok_or("Cant calculate mean")?
//             / calc_vector_euclidean_length(&features_variances.view());
//
//         let std = Array1::from_vec(stor_sqrt).mean().ok_or("Cant calc std")?;
//         let dencity = |cluster1: i32, cluster2: i32, point: &ArrayView1<f64>| -> usize {
//             let mut retval: usize = 0;
//             for i in [cluster1, cluster2] {
//                 for j in clusters[&i].iter() {
//                     let dist = find_euclidean_distance(point, &x.row(*j).view());
//                     retval += (dist <= std) as u8 as usize;
//                 }
//             }
//             retval
//         };
//         let mut acum: usize = 0;
//         for (i, c1) in clusters.keys().enumerate() {
//             for (j, c2) in clusters.keys().enumerate() {
//                 if i < j {
//                     let R = dencity(
//                         *c1,
//                         *c2,
//                         &((&cluster_centers[c1] + &cluster_centers[c2]) / 2.0).view(),
//                     ) / dencity(*c1, *c2, &cluster_centers[c1].view()).max(dencity(
//                         *c1,
//                         *c2,
//                         &cluster_centers[c2].view(),
//                     ));
//                     acum += R;
//                 }
//             }
//         }
//
//         let q = clusters.keys().len();
//         let Dbw = ((1 / (q * (q - 1))) * acum) as f64;
//
//         let value = S + Dbw;
//         Ok(value)
//     }
// }
