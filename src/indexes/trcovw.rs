use super::helpers::within_group_dispercion::WGDValue;
use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use ndarray::{ArrayView2, Axis};

#[derive(Clone, Copy, Debug)]
pub struct TrcovwIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(&self, wg: &ArrayView2<f64>) -> Result<f64, CalcError> {
        let var = wg.var_axis(Axis(0), 0.);
        let val = var.sum();
        Ok(val)
    }
}
pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, TrcovwIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, TrcovwIndexValue>) -> Self {
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
                .map(|val| TrcovwIndexValue { val }),
            Err(err) => Err(err),
        };
        self.sender.send_to_subscribers(res);
    }
}
