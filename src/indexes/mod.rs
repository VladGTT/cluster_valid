pub mod ball_hall;
// pub mod beale;
pub mod c_index;
pub mod calinski_harabasz;
pub mod davies_bouldin;
// pub mod duda;
pub mod dunn;
// pub mod gamma;
// pub mod gplus;
// pub mod mariott;
// pub mod mcclain;
// pub mod pseudot2;
// pub mod ptbiserial;
// pub mod ratkowsky;
// pub mod rubin;
// pub mod scott;
// pub mod sd;
// pub mod sdbw;
pub mod silhoutte;
// pub mod tau;
// pub mod tracew;
// pub mod trcovw;
use crate::calc_error::CalcError;
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    vec::Vec,
};
pub trait Subscriber<T> {
    fn recieve_data(&mut self, data: Arc<Result<T, CalcError>>);
}

#[derive(Default)]
pub struct Sender<'a, T: Send + Sync> {
    subscribers: Vec<Arc<Mutex<dyn Subscriber<T> + 'a + Send>>>,
}

impl<'a, T: Send + Sync + Clone> Sender<'a, T> {
    pub fn new(subscribers: Vec<Arc<Mutex<dyn Subscriber<T> + 'a + Send>>>) -> Self {
        Self { subscribers }
    }
    pub fn send_to_subscribers(&self, data: Arc<Result<T, CalcError>>) {
        self.subscribers.par_iter().for_each(|s| {
            let subscriber_lock = s.lock();
            match subscriber_lock {
                Ok(mut lock) => {
                    lock.recieve_data(data.clone());
                }
                Err(mut poison_error) => {
                    poison_error.get_mut().recieve_data(data.clone());
                }
            }
        })
    }
}
