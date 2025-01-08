use super::*;
#[derive(Default)]
pub struct Index;
impl Index {
    fn compute(
        &self,
        x: &ArrayView2<f64>,
        clusters_centroids: &HashMap<i32, Array1<f64>>,
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> Result<f64, CalcError> {
        let std = clusters
            .par_iter()
            .map(|(c, arr)| {
                arr.iter()
                    .map(|i| (&x.row(*i) - &clusters_centroids[c]).pow2().sum())
                    .sum::<f64>()
                    / arr.len() as f64
            })
            .sum::<f64>();
        let res = std / (clusters.keys().len() as f64);
        Ok(res)
    }
}

#[derive(Default)]
pub struct Node<'a> {
    index: Index,
    raw_data: Option<Arc<Result<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>), CalcError>>>,
    clusters: Option<Arc<Result<HashMap<i32, Array1<usize>>, CalcError>>>,
    clusters_centroids: Option<Arc<Result<HashMap<i32, Array1<f64>>, CalcError>>>,
    pub res: Option<Result<f64, CalcError>>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some(raw_data), Some(clusters), Some(clusters_centroids)) = (
            self.raw_data.as_ref(),
            self.clusters.as_ref(),
            self.clusters_centroids.as_ref(),
        ) {
            self.res = match (
                raw_data.as_ref(),
                clusters.as_ref(),
                clusters_centroids.as_ref(),
            ) {
                (Ok((x, _)), Ok(cls), Ok(cls_ctrds)) => Some(self.index.compute(x, cls_ctrds, cls)),
                (err1, err2, err3) => Some(Err(CalcError::from("Complete errror"))),
            };
        }
        // let combined_data = self
        //     .raw_data
        //     .combine(&self.clusters)
        //     .combine(&self.clusters_centroids);
        // self.res = match combined_data {
        //     CalcResult::Data((((x, _), clusters_centroids), clusters)) => {
        //         self.index.compute(x, clusters, clusters_centroids)
        //     }
        //     CalcResult::NoData => {
        //         return;
        //     }
        //     cd => cd.pass_err(),
        // };
    }
}

impl<'a> Subscriber<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>)> for Node<'a> {
    fn recieve_data(
        &mut self,
        data: Arc<Result<(&'a ArrayView2<'a, f64>, &'a ArrayView1<'a, i32>), CalcError>>,
    ) {
        self.raw_data = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<HashMap<i32, Array1<usize>>> for Node<'a> {
    fn recieve_data(&mut self, data: Arc<Result<HashMap<i32, Array1<usize>>, CalcError>>) {
        self.clusters = Some(data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<HashMap<i32, Array1<f64>>> for Node<'a> {
    fn recieve_data(&mut self, data: Arc<Result<HashMap<i32, Array1<f64>>, CalcError>>) {
        self.clusters_centroids = Some(data);
        self.process_when_ready();
    }
}
