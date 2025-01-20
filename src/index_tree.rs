use crate::indexes::ball_hall::BallHallIndexValue;
use crate::indexes::c_index::CIndexValue;
use crate::indexes::calinski_harabasz::CalinskiHarabaszIndexValue;
use crate::indexes::davies_bouldin::DaviesBouldinIndexValue;
use crate::indexes::dunn::DunnIndexValue;
use crate::indexes::silhoutte::SilhoutteIndexValue;
use crate::rust_ext::IndexTreeConfig;
use crate::{
    calc_error::CalcError,
    helpers::{
        clusters::ClustersNode, clusters_centroids::ClustersCentroidsNode,
        pairs_and_distances::PairsAndDistancesNode, raw_data::RawDataNode,
    },
    indexes::{
        ball_hall::Node as BallHallNode, c_index::Node as CIndexNode,
        calinski_harabasz::Node as CalinskiHarabaszNode, davies_bouldin::Node as DaviesBouldinNode,
        dunn::Node as DunnNode, silhoutte::Node as SilHoutteNode, Sender, Subscriber,
    },
};
use core::f64;
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
    pub davies_bouldin: Option<Result<DaviesBouldinIndexValue, CalcError>>,
    pub c_index: Option<Result<CIndexValue, CalcError>>,
    pub calinski_harabasz: Option<Result<CalinskiHarabaszIndexValue, CalcError>>,
    pub dunn: Option<Result<DunnIndexValue, CalcError>>,
    pub silhoutte: Option<Result<SilhoutteIndexValue, CalcError>>,
}

#[pymethods]
impl IndexTreeReturnValue {
    #[getter]
    fn get_ball_hall(&self) -> Result<Option<f64>, CalcError> {
        self.ball_hall.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_davies_bouldin(&self) -> Result<Option<f64>, CalcError> {
        self.davies_bouldin
            .clone()
            .map(|f| f.map(|v| v.val))
            .transpose()
    }
    #[getter]
    fn get_c_index(&self) -> Result<Option<f64>, CalcError> {
        self.c_index.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_calinski_harabasz(&self) -> Result<Option<f64>, CalcError> {
        self.calinski_harabasz
            .clone()
            .map(|f| f.map(|v| v.val))
            .transpose()
    }
    #[getter]
    fn get_dunn(&self) -> Result<Option<f64>, CalcError> {
        self.dunn.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_silhoutte(&self) -> Result<Option<f64>, CalcError> {
        self.silhoutte.clone().map(|f| f.map(|v| v.val)).transpose()
    }
}

impl Subscriber<BallHallIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Arc<Result<BallHallIndexValue, CalcError>>) {
        self.ball_hall = Some(data.as_ref().clone());
    }
}

impl Subscriber<DaviesBouldinIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Arc<Result<DaviesBouldinIndexValue, CalcError>>) {
        self.davies_bouldin = Some(data.as_ref().clone());
    }
}
impl Subscriber<CIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Arc<Result<CIndexValue, CalcError>>) {
        self.c_index = Some(data.as_ref().clone());
    }
}
impl Subscriber<CalinskiHarabaszIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Arc<Result<CalinskiHarabaszIndexValue, CalcError>>) {
        self.calinski_harabasz = Some(data.as_ref().clone());
    }
}
impl Subscriber<DunnIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Arc<Result<DunnIndexValue, CalcError>>) {
        self.dunn = Some(data.as_ref().clone());
    }
}
impl Subscriber<SilhoutteIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Arc<Result<SilhoutteIndexValue, CalcError>>) {
        self.silhoutte = Some(data.as_ref().clone());
    }
}
pub struct IndexTree<'a> {
    raw_data: RawDataNode<'a>,
    retval: Arc<Mutex<IndexTreeReturnValue>>,
}
impl<'a> IndexTree<'a> {
    pub fn new(config: &IndexTreeConfig) -> Self {
        let retval = Arc::new(Mutex::new(IndexTreeReturnValue::default()));
        let mut clusters_subscribers: Vec<
            Arc<Mutex<dyn Subscriber<HashMap<i32, Array1<usize>>> + Send>>,
        > = Vec::with_capacity(10);
        let mut clusters_centroids_subscribers: Vec<
            Arc<Mutex<dyn Subscriber<HashMap<i32, Array1<f64>>> + Send>>,
        > = Vec::with_capacity(10);
        let mut raw_data_subscribers: Vec<
            Arc<Mutex<dyn Subscriber<(&ArrayView2<f64>, &ArrayView1<i32>)> + Send>>,
        > = Vec::with_capacity(10);
        let mut pairs_and_distances_subscribers: Vec<
            Arc<Mutex<dyn Subscriber<(Vec<i8>, Vec<f64>)> + Send>>,
        > = Vec::with_capacity(10);

        if config.ball_hall {
            let ball_hall = Arc::new(Mutex::new(BallHallNode::new(Sender::new(vec![
                retval.clone()
            ]))));
            raw_data_subscribers.push(ball_hall.clone());
            clusters_subscribers.push(ball_hall.clone());
            clusters_centroids_subscribers.push(ball_hall.clone());
        }
        if config.silhoutte {
            let silhoutte = Arc::new(Mutex::new(SilHoutteNode::new(Sender::new(vec![
                retval.clone()
            ]))));
            raw_data_subscribers.push(silhoutte.clone());
            clusters_subscribers.push(silhoutte.clone());
        }
        if config.davies_bouldin {
            let davies_bouldin = Arc::new(Mutex::new(DaviesBouldinNode::new(Sender::new(vec![
                retval.clone(),
            ]))));
            raw_data_subscribers.push(davies_bouldin.clone());
            clusters_subscribers.push(davies_bouldin.clone());
            clusters_centroids_subscribers.push(davies_bouldin.clone());
        }
        if config.c_index {
            let c_index = Arc::new(Mutex::new(CIndexNode::new(Sender::new(vec![
                retval.clone()
            ]))));
            raw_data_subscribers.push(c_index.clone());
        }
        if config.calinski_harabasz {
            let calinski_harabasz =
                Arc::new(Mutex::new(CalinskiHarabaszNode::new(Sender::new(vec![
                    retval.clone(),
                ]))));
            raw_data_subscribers.push(calinski_harabasz.clone());
            clusters_subscribers.push(calinski_harabasz.clone());
            clusters_centroids_subscribers.push(calinski_harabasz.clone());
        }
        if config.dunn {
            let dunn = Arc::new(Mutex::new(DunnNode::new(Sender::new(vec![retval.clone()]))));
            raw_data_subscribers.push(dunn.clone());
        }

        if !pairs_and_distances_subscribers.is_empty() {
            let pairs_and_distances = Arc::new(Mutex::new(PairsAndDistancesNode::new(
                Sender::new(pairs_and_distances_subscribers),
            )));
            raw_data_subscribers.push(pairs_and_distances);
        }
        if !clusters_centroids_subscribers.is_empty() {
            let clusters_centroids = Arc::new(Mutex::new(ClustersCentroidsNode::new(Sender::new(
                clusters_centroids_subscribers,
            ))));
            clusters_subscribers.push(clusters_centroids.clone());
            raw_data_subscribers.push(clusters_centroids);
        }

        if !clusters_subscribers.is_empty() {
            let clusters = Arc::new(Mutex::new(ClustersNode::new(Sender::new(
                clusters_subscribers,
            ))));
            raw_data_subscribers.push(clusters);
        }

        let raw_data = RawDataNode::new(Sender::new(raw_data_subscribers));
        Self { raw_data, retval }
    }
    pub fn compute(
        self,
        data: (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>),
    ) -> IndexTreeReturnValue {
        self.raw_data.compute(data);
        match self.retval.lock() {
            Ok(lock) => lock.clone(),
            Err(poison_err) => poison_err.into_inner().clone(),
        }
    }
}
