use crate::{indexes::Subscribee, *};
pub type RawDataType<'a> = (&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>);
pub struct RawDataNode<'a> {
    data: RawDataType<'a>,
    pub subscribee: Subscribee<'a, RawDataType<'a>>,
}

impl<'a> RawDataNode<'a> {
    pub fn new(data: RawDataType<'a>, subscribee: Subscribee<'a, RawDataType<'a>>) -> Self {
        Self { data, subscribee }
    }
    pub fn compute(&mut self) {
        self.subscribee.send_to_subscribers(&self.data);
    }
}
