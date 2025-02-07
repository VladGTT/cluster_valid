use crate::{
    calc_error::CalcError,
    sender::{Sender, Subscriber},
};
use itertools::Itertools;
use ndarray::{ArcArray1, Array1, ArrayView1, ArrayView2};
#[derive(Default)]
pub struct Counts;
impl Counts {
    pub fn compute(&self, clusters: &ArrayView1<i32>) -> Result<ArcArray1<usize>, CalcError> {
        let counts = clusters.iter().counts();
        let vec = counts
            .into_iter()
            .sorted_by(|(a, _), (b, _)| a.cmp(b))
            .map(|(_, v)| v)
            .collect::<Vec<usize>>();
        let res = Array1::from_vec(vec);
        Ok(res.to_shared())
    }
}
pub struct CountsNode<'a> {
    index: Counts,
    sender: Sender<'a, ArcArray1<usize>>,
}
impl<'a> CountsNode<'a> {
    pub fn new(sender: Sender<'a, ArcArray1<usize>>) -> Self {
        Self {
            index: Counts,
            sender,
        }
    }
}

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for CountsNode<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        let res = match data {
            Ok((_, ref y)) => self.index.compute(y),
            Err(err) => Err(err),
        };
        self.sender.send_to_subscribers(res);
    }
}
