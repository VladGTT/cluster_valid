use super::*;
use helpers::raw_data::RawDataType;
#[derive(Default)]
pub struct Index;
impl Index {
    pub fn compute(&self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<f64, CalcError> {
        let n = y.len() * (y.len() - 1) / 2;
        let mut intercluster_distances: Vec<f64> = Vec::with_capacity(n);
        let mut intracluster_distances: Vec<f64> = Vec::with_capacity(n);

        for (i, (row1, cluster1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, cluster2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = (&row2 - &row1).pow2().sum().sqrt();
                    if cluster1 == cluster2 {
                        intracluster_distances.push(dist);
                    } else {
                        intercluster_distances.push(dist);
                    }
                }
            }
        }

        let max_intracluster = intracluster_distances
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .ok_or("Can't find max intracluster distance")?;
        let min_intercluster = intercluster_distances
            .iter()
            .min_by(|x, y| x.total_cmp(y))
            .ok_or("Can't find min intercluster distance")?;

        let value = min_intercluster / max_intracluster;
        Ok(value)
    }
}

#[derive(Default)]
pub struct Node {
    index: Index,
    pub res: Option<Result<f64, CalcError>>,
}

impl<'a> Subscriber<RawDataType<'a>> for Node {
    fn recieve_data(&mut self, data: &RawDataType<'a>) {
        let (x, y) = *data;
        self.res = Some(self.index.compute(x, y));
    }
}
