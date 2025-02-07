use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
#[derive(Clone, Copy, Debug)]
pub struct GammaIndexValue {
    pub val: f64,
}

#[derive(Default)]
pub struct Index;

impl Index {
    fn compute(&self, s_plus: usize, s_minus: usize) -> Result<f64, CalcError> {
        let value = (s_plus - s_minus) as f64 / (s_plus + s_minus) as f64;
        Ok(value)
    }
}
#[derive(Default)]
pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, GammaIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, GammaIndexValue>) -> Self {
        Self {
            index: Index,
            sender,
        }
    }
}
impl<'a> Subscriber<(usize, usize, usize)> for Node<'a> {
    fn recieve_data(&mut self, data: Result<(usize, usize, usize), CalcError>) {
        let res = match data.as_ref() {
            Ok((s_plus, s_minus, _)) => self
                .index
                .compute(*s_plus, *s_minus)
                .map(|val| GammaIndexValue { val }),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
