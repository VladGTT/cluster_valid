use crate::{
    calc_error::CalcError,
    sender::{Sender, Subscriber},
};
use ndarray::{ArcArray2, ArrayView1, ArrayView2, Axis};

#[derive(Clone, Debug)]
pub struct TDValue {
    pub val: ArcArray2<f64>,
}

#[derive(Default)]
pub struct TD;
impl TD {
    pub fn compute(&self, x: &ArrayView2<f64>) -> Result<ArcArray2<f64>, CalcError> {
        let data_center = x.mean_axis(Axis(0)).ok_or("Cant calc data centroid")?;
        let t = x - &data_center;
        let td = t.t().dot(&t);
        Ok(td.into_shared())
    }
}
pub struct TDNode<'a> {
    index: TD,
    sender: Sender<'a, TDValue>,
}
impl<'a> TDNode<'a> {
    pub fn new(sender: Sender<'a, TDValue>) -> Self {
        Self { index: TD, sender }
    }
}

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for TDNode<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        let res = match data {
            Ok((x, _)) => self.index.compute(&x).map(|val| TDValue { val }),
            Err(err) => Err(err),
        };
        self.sender.send_to_subscribers(res);
    }
}
