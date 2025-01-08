use crate::{indexes::Sender, *};
pub struct RawDataNode<'a> {
    data: (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>),
    pub sender: Sender<'a, (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)>,
}

impl<'a> RawDataNode<'a> {
    pub fn new(
        data: (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>),
        sender: Sender<'a, (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)>,
    ) -> Self {
        Self { data, sender }
    }
    pub fn compute(&mut self) {
        self.sender.send_to_subscribers(Arc::new(Ok(self.data)));
    }
}
