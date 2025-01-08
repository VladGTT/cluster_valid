pub mod ball_hall;
// pub mod beale;
// pub mod c_index;
// pub mod calinski_harabasz;
// pub mod davies_bouldin;
// pub mod duda;
// pub mod dunn;
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
// pub mod silhoutte;
// pub mod tau;
// pub mod tracew;
// pub mod trcovw;
use super::*;
use calc_error::CalcError;
use std::sync::Arc;
pub trait Subscriber<T> {
    fn recieve_data(&mut self, data: Arc<Result<T, CalcError>>);
}

pub struct Sender<'a, T: Send + Sync> {
    subscribers: Vec<&'a Mutex<dyn Subscriber<T> + 'a + Send>>,
}

impl<'a, T: Send + Sync> Default for Sender<'a, T> {
    fn default() -> Self {
        Self {
            subscribers: Vec::default(),
        }
    }
}
impl<'a, T: Send + Sync + Clone> Sender<'a, T> {
    pub fn new(subscribers: Vec<&'a Mutex<dyn Subscriber<T> + 'a + Send>>) -> Self {
        Self { subscribers }
    }
    // pub fn add_subscriber(&mut self)
    pub fn send_to_subscribers(&mut self, data: Arc<Result<T, CalcError>>) {
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
