use crate::indexes::ball_hall::BallHallIndexValue;
// use crate::indexes::c_index::CIndexValue;
// use crate::indexes::calinski_harabasz::CalinskiHarabaszIndexValue;
// use crate::indexes::davies_bouldin::DaviesBouldinIndexValue;
// use crate::indexes::dunn::DunnIndexValue;
// use crate::indexes::silhoutte::SilhoutteIndexValue;
use crate::{
    calc_error::CalcError,
    indexes::{
        ball_hall::Node as BallHallNode,
        // c_index::Node as CIndexNode,
        // calinski_harabasz::Node as CalinskiHarabaszNode, davies_bouldin::Node as DaviesBouldinNode,
        // dunn::Node as DunnNode, silhoutte::Node as SilHoutteNode,
        helpers::{
            clusters::ClustersNode, clusters_centroids::ClustersCentroidsNode,
            pairs_and_distances::PairsAndDistancesNode, raw_data::RawDataNode,
        },
    },
    sender::{Sender, Subscriber},
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use pyo3::{pyclass, pymethods};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[pyclass]
#[derive(Default, Debug, Clone)]
pub struct IndexTreeReturnValue {
    pub ball_hall: Option<Result<BallHallIndexValue, CalcError>>,
    // pub davies_bouldin: Option<Result<DaviesBouldinIndexValue, CalcError>>,
    // pub c_index: Option<Result<CIndexValue, CalcError>>,
    // pub calinski_harabasz: Option<Result<CalinskiHarabaszIndexValue, CalcError>>,
    // pub dunn: Option<Result<DunnIndexValue, CalcError>>,
    // pub silhoutte: Option<Result<SilhoutteIndexValue, CalcError>>,
}

#[pymethods]
impl IndexTreeReturnValue {
    #[getter]
    fn get_ball_hall(&self) -> Result<Option<f64>, CalcError> {
        self.ball_hall.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    // #[getter]
    // fn get_davies_bouldin(&self) -> Result<Option<f64>, CalcError> {
    //     self.davies_bouldin
    //         .clone()
    //         .map(|f| f.map(|v| v.val))
    //         .transpose()
    // }
    // #[getter]
    // fn get_c_index(&self) -> Result<Option<f64>, CalcError> {
    //     self.c_index.clone().map(|f| f.map(|v| v.val)).transpose()
    // }
    // #[getter]
    // fn get_calinski_harabasz(&self) -> Result<Option<f64>, CalcError> {
    //     self.calinski_harabasz
    //         .clone()
    //         .map(|f| f.map(|v| v.val))
    //         .transpose()
    // }
    // #[getter]
    // fn get_dunn(&self) -> Result<Option<f64>, CalcError> {
    //     self.dunn.clone().map(|f| f.map(|v| v.val)).transpose()
    // }
    // #[getter]
    // fn get_silhoutte(&self) -> Result<Option<f64>, CalcError> {
    //     self.silhoutte.clone().map(|f| f.map(|v| v.val)).transpose()
    // }
}

impl Subscriber<BallHallIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<BallHallIndexValue, CalcError>) {
        self.ball_hall = Some(data);
    }
}

// impl Subscriber<DaviesBouldinIndexValue> for IndexTreeReturnValue {
//     fn recieve_data(&mut self, data: Arc<Result<DaviesBouldinIndexValue, CalcError>>) {
//         self.davies_bouldin = Some(data.as_ref().clone());
//     }
// }
// impl Subscriber<CIndexValue> for IndexTreeReturnValue {
//     fn recieve_data(&mut self, data: Arc<Result<CIndexValue, CalcError>>) {
//         self.c_index = Some(data.as_ref().clone());
//     }
// }
// impl Subscriber<CalinskiHarabaszIndexValue> for IndexTreeReturnValue {
//     fn recieve_data(&mut self, data: Arc<Result<CalinskiHarabaszIndexValue, CalcError>>) {
//         self.calinski_harabasz = Some(data.as_ref().clone());
//     }
// }
// impl Subscriber<DunnIndexValue> for IndexTreeReturnValue {
//     fn recieve_data(&mut self, data: Arc<Result<DunnIndexValue, CalcError>>) {
//         self.dunn = Some(data.as_ref().clone());
//     }
// }
// impl Subscriber<SilhoutteIndexValue> for IndexTreeReturnValue {
//     fn recieve_data(&mut self, data: Arc<Result<SilhoutteIndexValue, CalcError>>) {
//         self.silhoutte = Some(data.as_ref().clone());
//     }
// }
pub struct IndexTree<'a> {
    raw_data: RawDataNode<'a>,
    retval: Arc<Mutex<IndexTreeReturnValue>>,
}
impl<'a> IndexTree<'a> {
    pub fn compute(self, data: (ArrayView2<'a, f64>, ArrayView1<'a, i32>)) -> IndexTreeReturnValue {
        self.raw_data.compute(data);
        match self.retval.lock() {
            Ok(lock) => lock.clone(),
            Err(poison_err) => poison_err.into_inner().clone(),
        }
    }
}

#[derive(Default)]
pub struct IndexTreeBuilder<'a> {
    retval: Arc<Mutex<IndexTreeReturnValue>>,
    clusters_sender: Sender<'a, Arc<HashMap<i32, Array1<usize>>>>,
    clusters_centroids_sender: Sender<'a, Arc<HashMap<i32, Array1<f64>>>>,
    raw_data_sender: Sender<'a, (ArrayView2<'a, f64>, ArrayView1<'a, i32>)>,
    pairs_and_distances_sender: Sender<'a, Arc<(Vec<i8>, Vec<f64>)>>,
}

impl<'a> IndexTreeBuilder<'a> {
    pub fn add_ball_hall(mut self) -> Self {
        let ball_hall = Arc::new(Mutex::new(BallHallNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(ball_hall.clone());
        self.clusters_sender.add_subscriber(ball_hall.clone());
        self.clusters_centroids_sender
            .add_subscriber(ball_hall.clone());
        self
    }
    // pub fn add_silhoutte(mut self) -> Self {
    //     let silhoutte = Arc::new(Mutex::new(SilHoutteNode::new(Sender::new(vec![self
    //         .retval
    //         .clone()]))));
    //     self.raw_data_subscribers.push(silhoutte.clone());
    //     self.clusters_subscribers.push(silhoutte.clone());
    //     self
    // }
    // pub fn add_davies_bouldin(mut self) -> Self {
    //     let davies_bouldin = Arc::new(Mutex::new(DaviesBouldinNode::new(Sender::new(vec![self
    //         .retval
    //         .clone()]))));
    //     self.raw_data_subscribers.push(davies_bouldin.clone());
    //
    //     self.clusters_subscribers.push(davies_bouldin.clone());
    //     self.clusters_centroids_subscribers
    //         .push(davies_bouldin.clone());
    //     self
    // }
    // pub fn add_calinski_harabasz(mut self) -> Self {
    //     let calinski_harabasz = Arc::new(Mutex::new(CalinskiHarabaszNode::new(Sender::new(vec![
    //         self.retval.clone(),
    //     ]))));
    //     self.raw_data_subscribers.push(calinski_harabasz.clone());
    //
    //     self.clusters_subscribers.push(calinski_harabasz.clone());
    //     self.clusters_centroids_subscribers
    //         .push(calinski_harabasz.clone());
    //     self
    // }
    // pub fn add_c_index(mut self) -> Self {
    //     let c_index = Arc::new(Mutex::new(CIndexNode::new(Sender::new(vec![self
    //         .retval
    //         .clone()]))));
    //     self.raw_data_subscribers.push(c_index.clone());
    //     self
    // }
    // pub fn add_dunn(mut self) -> Self {
    //     let dunn = Arc::new(Mutex::new(DunnNode::new(Sender::new(vec![self
    //         .retval
    //         .clone()]))));
    //     self.raw_data_subscribers.push(dunn.clone());
    //     self
    // }
    pub fn finish(mut self) -> IndexTree<'a> {
        if !self.pairs_and_distances_sender.is_empty() {
            let pairs_and_distances = Arc::new(Mutex::new(PairsAndDistancesNode::new(
                self.pairs_and_distances_sender,
            )));
            self.raw_data_sender.add_subscriber(pairs_and_distances);
        }
        if !self.clusters_centroids_sender.is_empty() {
            let clusters_centroids = Arc::new(Mutex::new(ClustersCentroidsNode::new(
                self.clusters_centroids_sender,
            )));
            self.clusters_sender
                .add_subscriber(clusters_centroids.clone());
            self.raw_data_sender.add_subscriber(clusters_centroids);
        }

        if !self.clusters_sender.is_empty() {
            let clusters = Arc::new(Mutex::new(ClustersNode::new(self.clusters_sender)));
            self.raw_data_sender.add_subscriber(clusters);
        }

        let raw_data = RawDataNode::new(self.raw_data_sender);
        IndexTree {
            raw_data,
            retval: self.retval,
        }
    }
}
