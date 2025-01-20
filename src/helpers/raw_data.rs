use crate::indexes::Sender;
use ndarray::{ArrayView1, ArrayView2};
use std::sync::Arc;
pub struct RawDataNode<'a> {
    pub sender: Sender<'a, (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)>,
}

impl<'a> RawDataNode<'a> {
    pub fn new(sender: Sender<'a, (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)>) -> Self {
        Self { sender }
    }
    pub fn compute(&self, data: (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)) {
        self.sender.send_to_subscribers(Arc::new(Ok(data)));
    }
}
