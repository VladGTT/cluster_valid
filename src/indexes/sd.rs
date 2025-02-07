use crate::calc_error::{CalcError, CombineErrors};
use ndarray::{ArcArray1, ArcArray2, ArrayView1, ArrayView2};

use crate::sender::{Sender, Subscriber};

use super::helpers::{clusters_centroids::ClustersCentroidsValue, scat::ScatValue};

#[derive(Clone, Copy, Debug)]
pub struct SDIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        scat: &f64,
        clusters_centroids: &ArrayView2<f64>,
    ) -> Result<f64, CalcError> {
        let mut d = 0.0;
        let mut d_max = f64::MIN;
        let mut d_min = f64::MAX;
        for (i, row1) in clusters_centroids.rows().into_iter().enumerate() {
            let mut dist_acum = 0.0;
            for (j, row2) in clusters_centroids.rows().into_iter().enumerate() {
                if i != j {
                    let dist = (&row2 - &row1).pow2().sum().sqrt();
                    dist_acum += dist;
                    if i < j {
                        if dist > d_max {
                            d_max = dist;
                        }
                        if dist < d_min {
                            d_min = dist;
                        }
                    }
                }
            }
            if dist_acum != 0.0 {
                d += 1. / dist_acum;
            }
        }
        let value = d * d_max / d_min;

        Ok(value)
    }
}

pub struct Node<'a> {
    index: Index,
    scat: Option<Result<(f64, ArcArray1<f64>, f64), CalcError>>,
    clusters_centroids: Option<Result<ArcArray2<f64>, CalcError>>,
    sender: Sender<'a, SDIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(scat), Some(clusters_centroids)) =
            (self.scat.as_ref(), self.clusters_centroids.as_ref())
        {
            let res = match scat.combine(clusters_centroids) {
                Ok(((val, _, _), cls_ctrds)) => self
                    .index
                    .compute(val, &cls_ctrds.view())
                    .map(|val| SDIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.scat = None;
            self.clusters_centroids = None;
        }
    }
    pub fn new(sender: Sender<'a, SDIndexValue>) -> Self {
        Self {
            index: Index,
            scat: None,
            clusters_centroids: None,
            sender,
        }
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
// impl Computable for IndexDis {
//     fn compute(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, CalcError> {
//         let mut d = 0.0;
//         let mut d_max = f64::MIN;
//         let mut d_min = f64::MAX;
//         let clusters = calc_clusters(&y);
//         let cluster_centroids = calc_clusters_centers(&clusters, &x);
//
//         for (i, row1) in &cluster_centroids {
//             let mut dist_acum = 0.0;
//             for (j, row2) in &cluster_centroids {
//                 if i != j {
//                     let dist = find_euclidean_distance(&row1.view(), &row2.view());
//                     dist_acum += dist;
//                     if i < j {
//                         if dist > d_max {
//                             d_max = dist;
//                         }
//                         if dist < d_min {
//                             d_min = dist;
//                         }
//                     }
//                 }
//             }
//             if dist_acum != 0.0 {
//                 d += 1. / dist_acum;
//             }
//         }
//         let value = d * d_max / d_min;
//         Ok(value)
//     }
// }
