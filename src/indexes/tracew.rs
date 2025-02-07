use super::helpers::within_group_dispercion::WGDValue;
use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use ndarray::ArrayView2;

#[derive(Clone, Copy, Debug)]
pub struct TracewIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(&self, wg: &ArrayView2<f64>) -> Result<f64, CalcError> {
        Ok(wg.diag().sum())
    }
}
pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, TracewIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, TracewIndexValue>) -> Self {
        Self {
            index: Index,
            sender,
        }
    }
}

impl<'a> Subscriber<WGDValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<WGDValue, CalcError>) {
        let res = match data.map(|v| v.val) {
            Ok(wg) => self
                .index
                .compute(&wg.view())
                .map(|val| TracewIndexValue { val }),
            Err(err) => Err(err),
        };
        self.sender.send_to_subscribers(res);
    }
}
