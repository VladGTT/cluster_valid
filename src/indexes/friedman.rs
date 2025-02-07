use super::helpers::{between_group_dispercion::BGDValue, within_group_dispercion::WGDValue};
use crate::calc_error::{CalcError, CombineErrors};
use crate::sender::{Sender, Subscriber};
use ndarray::{ArcArray2, ArrayView2};
use ndarray_linalg::Inverse;

#[derive(Clone, Copy, Debug)]
pub struct FriedmanIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(&self, wg: &ArrayView2<f64>, bg: &ArrayView2<f64>) -> Result<f64, CalcError> {
        let wg_inv = wg.inv().map_err(|e| CalcError::from(format!("{e:?}")))?;
        let value = wg_inv.dot(bg).diag().sum();
        Ok(value)
    }
}
pub struct Node<'a> {
    index: Index,
    wg: Option<Result<ArcArray2<f64>, CalcError>>,
    bg: Option<Result<ArcArray2<f64>, CalcError>>,
    sender: Sender<'a, FriedmanIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(wg), Some(bg)) = (self.wg.as_ref(), self.bg.as_ref()) {
            let res = match wg.combine(bg) {
                Ok((wg, bg)) => self
                    .index
                    .compute(&wg.view(), &bg.view())
                    .map(|val| FriedmanIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.wg = None;
            self.bg = None;
        }
    }
    pub fn new(sender: Sender<'a, FriedmanIndexValue>) -> Self {
        Self {
            index: Index,
            wg: None,
            bg: None,
            sender,
        }
    }
}

impl<'a> Subscriber<WGDValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<WGDValue, CalcError>) {
        self.wg = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}

impl<'a> Subscriber<BGDValue> for Node<'a> {
    fn recieve_data(&mut self, data: Result<BGDValue, CalcError>) {
        self.bg = Some(data.map(|v| v.val));
        self.process_when_ready();
    }
}
