use crate::sender::Sender;
use ndarray::{ArrayView1, ArrayView2};
pub struct RawDataNode<'a> {
    pub sender: Sender<'a, (ArrayView2<'a, f64>, ArrayView1<'a, i32>)>,
}

impl<'a> RawDataNode<'a> {
    pub fn new(sender: Sender<'a, (ArrayView2<'a, f64>, ArrayView1<'a, i32>)>) -> Self {
        Self { sender }
    }
    pub fn compute(&self, data: (ArrayView2<'a, f64>, ArrayView1<'a, i32>)) {
        self.sender.send_to_subscribers(Ok(data));
    }
}
