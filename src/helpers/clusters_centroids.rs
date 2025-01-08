use calc_error::CalcError;

use crate::{
    indexes::{Sender, Subscriber},
    *,
};
#[derive(Default)]
pub struct ClustersCentroids;
impl ClustersCentroids {
    pub fn compute(
        &self,
        clusters_indexes: &HashMap<i32, Array1<usize>>,
        data: &ArrayView2<f64>,
    ) -> Result<HashMap<i32, Array1<f64>>, CalcError> {
        let res = clusters_indexes
            .into_par_iter()
            .map(|(c, arr)| {
                let mut sum: Array1<f64> = Array1::zeros(data.ncols());
                for i in arr {
                    sum += &data.row(*i);
                }
                (*c, sum / arr.len() as f64)
            })
            .collect::<HashMap<i32, Array1<f64>>>();
        Ok(res)
    }
}
pub struct ClustersCentroidsNode<'a> {
    index: ClustersCentroids,
    clusters: Option<Arc<Result<HashMap<i32, Array1<usize>>, CalcError>>>,
    raw_data: Option<Arc<Result<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>), CalcError>>>,
    sender: Sender<'a, HashMap<i32, Array1<f64>>>,
}
impl<'a> ClustersCentroidsNode<'a> {
    pub fn with_sender(sender: Sender<'a, HashMap<i32, Array1<f64>>>) -> Self {
        Self {
            index: ClustersCentroids,
            clusters: None,
            raw_data: None,
            sender,
        }
    }
    fn process_when_ready(&mut self) {
        if let (Some(clusters), Some(raw_data)) = (self.clusters.as_ref(), self.raw_data.as_ref()) {
            let res = match (clusters.as_ref(), raw_data.as_ref()) {
                (Ok(cls), Ok((x, _))) => self.index.compute(cls, x),
                (Err(cls_err), Ok(_)) => Err(cls_err.clone()),
                (Ok(_), Err(rd_err)) => Err(rd_err.clone()),
                _ => Err(CalcError::from("all wrong")),
            };
            self.sender.send_to_subscribers(Arc::new(res));
        }
    }
}
impl<'a> Subscriber<HashMap<i32, Array1<usize>>> for ClustersCentroidsNode<'a> {
    fn recieve_data(&mut self, data: Arc<Result<HashMap<i32, Array1<usize>>, CalcError>>) {
        self.clusters = Some(data);
        self.process_when_ready();
    }
}

impl<'a> Subscriber<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)>
    for ClustersCentroidsNode<'a>
{
    fn recieve_data(
        &mut self,
        data: Arc<Result<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>), CalcError>>,
    ) {
        self.raw_data = Some(data);
        self.process_when_ready();
    }
}
