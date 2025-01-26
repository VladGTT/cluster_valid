use crate::indexes::ball_hall::BallHallIndexValue;
use crate::indexes::c_index::CIndexValue;
use crate::indexes::calinski_harabasz::CalinskiHarabaszIndexValue;
use crate::indexes::davies_bouldin::DaviesBouldinIndexValue;
use crate::indexes::dunn::DunnIndexValue;
use crate::indexes::friedman::FriedmanIndexValue;
use crate::indexes::gamma::GammaIndexValue;
use crate::indexes::gplus::GplusIndexValue;
use crate::indexes::mariott::MariottIndexValue;
use crate::indexes::mcclain::McclainIndexValue;
use crate::indexes::ptbiserial::PtbiserialIndexValue;
use crate::indexes::ratkowsky::RatkowskyIndexValue;
use crate::indexes::rubin::RubinIndexValue;
use crate::indexes::scott::ScottIndexValue;
use crate::indexes::silhouette::SilhouetteIndexValue;
use crate::indexes::tau::TauIndexValue;
use crate::indexes::tracew::TracewIndexValue;

use crate::{
    calc_error::CalcError,
    indexes::{
        ball_hall::Node as BallHallNode,
        c_index::Node as CIndexNode,
        calinski_harabasz::Node as CalinskiHarabaszNode,
        davies_bouldin::Node as DaviesBouldinNode,
        dunn::Node as DunnNode,
        friedman::Node as FriedmanNode,
        gamma::Node as GammaNode,
        gplus::Node as GplusNode,
        helpers::{
            clusters::ClustersNode, clusters_centroids::ClustersCentroidsNode,
            pairs_and_distances::PairsAndDistancesNode, raw_data::RawDataNode,
        },
        mariott::Node as MariottNode,
        mcclain::Node as McclainNode,
        ptbiserial::Node as PtbiserialNode,
        ratkowsky::Node as RatkowskyNode,
        rubin::Node as RubinNode,
        scott::Node as ScottNode,
        silhouette::Node as SilhouetteNode,
        tau::Node as TauNode,
        tracew::Node as TracewNode,
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
    pub davies_bouldin: Option<Result<DaviesBouldinIndexValue, CalcError>>,
    pub c_index: Option<Result<CIndexValue, CalcError>>,
    pub calinski_harabasz: Option<Result<CalinskiHarabaszIndexValue, CalcError>>,
    pub dunn: Option<Result<DunnIndexValue, CalcError>>,
    pub silhouette: Option<Result<SilhouetteIndexValue, CalcError>>,
    pub rubin: Option<Result<RubinIndexValue, CalcError>>,
    pub mariott: Option<Result<MariottIndexValue, CalcError>>,
    pub scott: Option<Result<ScottIndexValue, CalcError>>,
    pub friedman: Option<Result<FriedmanIndexValue, CalcError>>,
    pub tau: Option<Result<TauIndexValue, CalcError>>,
    pub gamma: Option<Result<GammaIndexValue, CalcError>>,
    pub gplus: Option<Result<GplusIndexValue, CalcError>>,
    pub tracew: Option<Result<TracewIndexValue, CalcError>>,
    pub mcclain: Option<Result<McclainIndexValue, CalcError>>,
    pub ptbiserial: Option<Result<PtbiserialIndexValue, CalcError>>,
    pub ratkowsky: Option<Result<RatkowskyIndexValue, CalcError>>,
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
    fn get_silhouette(&self) -> Result<Option<f64>, CalcError> {
        self.silhouette
            .clone()
            .map(|f| f.map(|v| v.val))
            .transpose()
    }
    #[getter]
    fn get_rubin(&self) -> Result<Option<f64>, CalcError> {
        self.rubin.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_mariott(&self) -> Result<Option<f64>, CalcError> {
        self.mariott.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_scott(&self) -> Result<Option<f64>, CalcError> {
        self.scott.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_friedman(&self) -> Result<Option<f64>, CalcError> {
        self.friedman.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_tau(&self) -> Result<Option<f64>, CalcError> {
        self.friedman.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_gamma(&self) -> Result<Option<f64>, CalcError> {
        self.gamma.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_gplus(&self) -> Result<Option<f64>, CalcError> {
        self.gplus.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_tracew(&self) -> Result<Option<f64>, CalcError> {
        self.tracew.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_mcclain(&self) -> Result<Option<f64>, CalcError> {
        self.mcclain.clone().map(|f| f.map(|v| v.val)).transpose()
    }
    #[getter]
    fn get_ptbiserial(&self) -> Result<Option<f64>, CalcError> {
        self.ptbiserial
            .clone()
            .map(|f| f.map(|v| v.val))
            .transpose()
    }
    #[getter]
    fn get_ratkowsky(&self) -> Result<Option<f64>, CalcError> {
        self.ratkowsky.clone().map(|f| f.map(|v| v.val)).transpose()
    }
}

impl Subscriber<BallHallIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<BallHallIndexValue, CalcError>) {
        self.ball_hall = Some(data);
    }
}

impl Subscriber<DaviesBouldinIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<DaviesBouldinIndexValue, CalcError>) {
        self.davies_bouldin = Some(data);
    }
}
impl Subscriber<CIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<CIndexValue, CalcError>) {
        self.c_index = Some(data);
    }
}
impl Subscriber<CalinskiHarabaszIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<CalinskiHarabaszIndexValue, CalcError>) {
        self.calinski_harabasz = Some(data);
    }
}
impl Subscriber<DunnIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<DunnIndexValue, CalcError>) {
        self.dunn = Some(data);
    }
}
impl Subscriber<SilhouetteIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<SilhouetteIndexValue, CalcError>) {
        self.silhouette = Some(data);
    }
}
impl Subscriber<RubinIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<RubinIndexValue, CalcError>) {
        self.rubin = Some(data);
    }
}
impl Subscriber<MariottIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<MariottIndexValue, CalcError>) {
        self.mariott = Some(data);
    }
}
impl Subscriber<ScottIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<ScottIndexValue, CalcError>) {
        self.scott = Some(data);
    }
}
impl Subscriber<FriedmanIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<FriedmanIndexValue, CalcError>) {
        self.friedman = Some(data);
    }
}
impl Subscriber<TauIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<TauIndexValue, CalcError>) {
        self.tau = Some(data);
    }
}
impl Subscriber<GammaIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<GammaIndexValue, CalcError>) {
        self.gamma = Some(data);
    }
}
impl Subscriber<GplusIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<GplusIndexValue, CalcError>) {
        self.gplus = Some(data);
    }
}
impl Subscriber<TracewIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<TracewIndexValue, CalcError>) {
        self.tracew = Some(data);
    }
}
impl Subscriber<McclainIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<McclainIndexValue, CalcError>) {
        self.mcclain = Some(data);
    }
}
impl Subscriber<PtbiserialIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<PtbiserialIndexValue, CalcError>) {
        self.ptbiserial = Some(data);
    }
}
impl Subscriber<RatkowskyIndexValue> for IndexTreeReturnValue {
    fn recieve_data(&mut self, data: Result<RatkowskyIndexValue, CalcError>) {
        self.ratkowsky = Some(data);
    }
}
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
    pub fn add_silhouette(mut self) -> Self {
        let silhouette = Arc::new(Mutex::new(SilhouetteNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(silhouette.clone());
        self.clusters_sender.add_subscriber(silhouette.clone());
        self
    }
    pub fn add_davies_bouldin(mut self) -> Self {
        let davies_bouldin = Arc::new(Mutex::new(DaviesBouldinNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(davies_bouldin.clone());

        self.clusters_sender.add_subscriber(davies_bouldin.clone());
        self.clusters_centroids_sender
            .add_subscriber(davies_bouldin.clone());
        self
    }
    pub fn add_calinski_harabasz(mut self) -> Self {
        let calinski_harabasz = Arc::new(Mutex::new(CalinskiHarabaszNode::new(Sender::new(vec![
            self.retval.clone(),
        ]))));
        self.raw_data_sender
            .add_subscriber(calinski_harabasz.clone());

        self.clusters_sender
            .add_subscriber(calinski_harabasz.clone());
        self.clusters_centroids_sender
            .add_subscriber(calinski_harabasz.clone());
        self
    }
    pub fn add_c_index(mut self) -> Self {
        let c_index = Arc::new(Mutex::new(CIndexNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(c_index.clone());
        self
    }
    pub fn add_dunn(mut self) -> Self {
        let dunn = Arc::new(Mutex::new(DunnNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(dunn.clone());
        self
    }
    pub fn add_rubin(mut self) -> Self {
        let rubin = Arc::new(Mutex::new(RubinNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(rubin.clone());

        self.clusters_centroids_sender.add_subscriber(rubin.clone());
        self
    }
    pub fn add_mariott(mut self) -> Self {
        let mariott = Arc::new(Mutex::new(MariottNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(mariott.clone());

        self.clusters_centroids_sender
            .add_subscriber(mariott.clone());
        self
    }
    pub fn add_scott(mut self) -> Self {
        let scott = Arc::new(Mutex::new(ScottNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(scott.clone());

        self.clusters_centroids_sender.add_subscriber(scott.clone());
        self
    }
    pub fn add_friedman(mut self) -> Self {
        let friedman = Arc::new(Mutex::new(FriedmanNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(friedman.clone());

        self.clusters_sender.add_subscriber(friedman.clone());
        self.clusters_centroids_sender
            .add_subscriber(friedman.clone());
        self
    }
    pub fn add_tau(mut self) -> Self {
        let tau = Arc::new(Mutex::new(TauNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(tau.clone());

        self.pairs_and_distances_sender.add_subscriber(tau.clone());
        self
    }
    pub fn add_gamma(mut self) -> Self {
        let gamma = Arc::new(Mutex::new(GammaNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.pairs_and_distances_sender.add_subscriber(gamma);
        self
    }
    pub fn add_gplus(mut self) -> Self {
        let gplus = Arc::new(Mutex::new(GplusNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(gplus.clone());
        self.pairs_and_distances_sender.add_subscriber(gplus);
        self
    }
    pub fn add_tracew(mut self) -> Self {
        let tracew = Arc::new(Mutex::new(TracewNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(tracew.clone());
        self.clusters_centroids_sender.add_subscriber(tracew);
        self
    }
    pub fn add_mcclain(mut self) -> Self {
        let mcclain = Arc::new(Mutex::new(McclainNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(mcclain);
        self
    }
    pub fn add_ptbiserial(mut self) -> Self {
        let ptbiserial = Arc::new(Mutex::new(PtbiserialNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(ptbiserial);
        self
    }
    pub fn add_ratkowsky(mut self) -> Self {
        let ratkowsky = Arc::new(Mutex::new(RatkowskyNode::new(Sender::new(vec![self
            .retval
            .clone()]))));
        self.raw_data_sender.add_subscriber(ratkowsky.clone());

        self.clusters_sender.add_subscriber(ratkowsky.clone());
        self.clusters_centroids_sender.add_subscriber(ratkowsky);
        self
    }
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
