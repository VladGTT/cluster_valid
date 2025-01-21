use crate::calc_error::CalcError;
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    vec::Vec,
};
pub trait Subscriber<T> {
    fn recieve_data(&mut self, data: Result<T, CalcError>);
}

pub struct Sender<'a, T: Send> {
    subscribers: Vec<Arc<Mutex<dyn Subscriber<T> + 'a + Send>>>,
}

impl<'a, T: Send> Default for Sender<'a, T> {
    fn default() -> Self {
        Self {
            subscribers: Vec::with_capacity(5),
        }
    }
}

impl<'a, T: Send + Sync + Clone> Sender<'a, T> {
    pub fn new(subscribers: Vec<Arc<Mutex<dyn Subscriber<T> + 'a + Send>>>) -> Self {
        Self { subscribers }
    }
    pub fn add_subscriber(&mut self, value: Arc<Mutex<dyn Subscriber<T> + 'a + Send>>) {
        self.subscribers.push(value);
    }
    pub fn is_empty(&self) -> bool {
        self.subscribers.is_empty()
    }
    pub fn send_to_subscribers(&self, data: Result<T, CalcError>) {
        self.subscribers.par_iter().for_each(|s| match s.lock() {
            Ok(mut lock) => {
                lock.recieve_data(data.clone());
            }
            Err(mut poison_error) => {
                poison_error.get_mut().recieve_data(data.clone());
            }
        })
    }
}
