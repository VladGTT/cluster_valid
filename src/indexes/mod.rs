pub mod ball_hall;
// pub mod beale;
pub mod c_index;
pub mod calinski_harabasz;
pub mod davies_bouldin;
// pub mod duda;
pub mod dunn;
pub mod gamma;
pub mod gplus;
pub mod mariott;
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

use super::*;
pub trait Subscriber<T> {
    fn recieve_data(&mut self, data: &T);
}

pub struct Subscribee<'a, T: Send + Sync> {
    subscribers: Vec<&'a Mutex<dyn Subscriber<T> + 'a + Send>>,
}

impl<'a, T: Send + Sync> Default for Subscribee<'a, T> {
    fn default() -> Self {
        Subscribee {
            subscribers: Vec::default(),
        }
    }
}
impl<'a, T: Send + Sync> Subscribee<'a, T> {
    pub fn new(subscribers: Vec<&'a Mutex<dyn Subscriber<T> + 'a + Send>>) -> Self {
        Subscribee { subscribers }
    }

    pub fn send_to_subscribers(&mut self, data: &T) {
        self.subscribers
            .par_iter()
            .for_each(|s| s.lock().unwrap().recieve_data(data))
    }
}
