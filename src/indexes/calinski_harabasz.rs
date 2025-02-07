use super::helpers::{between_group_dispercion::BGDValue, within_group_dispercion::WGDValue};
use crate::calc_error::{CalcError, CombineErrors};
use crate::sender::{Sender, Subscriber};
use ndarray::{ArcArray1, ArcArray2, ArrayView1, ArrayView2};

#[derive(Clone, Copy, Debug)]
pub struct CalinskiHarabaszIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(
        &self,
        wg: &ArrayView2<f64>,
        bg: &ArrayView2<f64>,
        counts: &ArrayView1<usize>,
    ) -> Result<f64, CalcError> {
        let trace_wg = wg.diag().sum();
        let trace_bg = bg.diag().sum();
        let q = counts.len() as f64;
        let n = counts.sum() as f64;
        let val = (trace_bg / (q - 1.)) * ((n - q) / trace_wg);
        Ok(val)
    }
}
pub struct Node<'a> {
    index: Index,
    wg: Option<Result<ArcArray2<f64>, CalcError>>,
    bg: Option<Result<ArcArray2<f64>, CalcError>>,
    counts: Option<Result<ArcArray1<usize>, CalcError>>,
    sender: Sender<'a, CalinskiHarabaszIndexValue>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(wg), Some(bg), Some(counts)) =
            (self.wg.as_ref(), self.bg.as_ref(), self.counts.as_ref())
        {
            let res = match wg.combine(bg).combine(counts) {
                Ok(((wg, bg), cnts)) => self
                    .index
                    .compute(&wg.view(), &bg.view(), &cnts.view())
                    .map(|val| CalinskiHarabaszIndexValue { val }),
                Err(err) => Err(err),
            };
            self.sender.send_to_subscribers(res);
            self.wg = None;
            self.bg = None;
            self.counts = None;
        }
    }
    pub fn new(sender: Sender<'a, CalinskiHarabaszIndexValue>) -> Self {
        Self {
            index: Index,
            bg: None,
            wg: None,
            counts: None,
            sender,
        }
    }
}

impl<'a> Subscriber<ArcArray1<usize>> for Node<'a> {
    fn recieve_data(&mut self, data: Result<ArcArray1<usize>, CalcError>) {
        self.counts = Some(data);
        self.process_when_ready();
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
