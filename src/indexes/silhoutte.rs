use super::*;
use calc_error::CalcError;
use helpers::{clusters::ClustersType, raw_data::RawDataType};
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(
        &self,
        x: &ArrayView2<f64>,
        clusters: &HashMap<i32, Array1<usize>>,
    ) -> Result<f64, CalcError> {
        let mut temp: Vec<f64> = Vec::with_capacity(clusters.keys().len() - 1);
        let mut stor: Vec<f64> = Vec::with_capacity(x.nrows());
        for (c, arr) in clusters.iter() {
            for i in arr {
                let mut sum_inter_dists = 0.0;
                let row = x.row(*i);
                for j in arr {
                    if i != j {
                        sum_inter_dists += (&x.row(*j) - &row).pow2().sum().sqrt();
                    }
                }
                for (c2, arr2) in clusters.iter() {
                    if c2 != c {
                        let mut sum_intra_dists = 0.0;
                        for j2 in arr2 {
                            sum_intra_dists += (&x.row(*j2) - &row).pow2().sum().sqrt();
                        }
                        temp.push(sum_intra_dists / arr2.len() as f64);
                    }
                }
                let a = sum_inter_dists
                    / if arr.len() == 1 {
                        1.0
                    } else {
                        (arr.len() - 1) as f64
                    };
                let b = temp
                    .iter()
                    .min_by(|a, b| a.total_cmp(b))
                    .ok_or("Cant find min")?;

                stor.push((b - a) / a.max(*b));
                temp.clear()
            }
        }
        let value = Array1::from_vec(stor).mean().ok_or("Cant calc mean")?;
        Ok(value)
    }
}

#[derive(Default)]
pub struct Node<'a> {
    index: Index,
    raw_data: Option<RawDataType<'a>>,
    clusters: Option<ClustersType>,
    pub res: Option<Result<f64, CalcError>>,
}

impl<'a> Node<'a> {
    fn process_when_ready(&mut self) {
        if let (Some((x, _)), Some(clusters)) = (self.raw_data.as_ref(), self.clusters.as_ref()) {
            self.res = Some(self.index.compute(x, clusters));
        }
    }
}
impl<'a> Subscriber<RawDataType<'a>> for Node<'a> {
    fn recieve_data(&mut self, data: &RawDataType<'a>) {
        self.raw_data = Some(*data);
        self.process_when_ready();
    }
}
impl<'a> Subscriber<ClustersType> for Node<'a> {
    fn recieve_data(&mut self, data: &ClustersType) {
        self.clusters = Some(data.clone());
        self.process_when_ready();
    }
}
