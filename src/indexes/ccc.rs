use crate::calc_error::CalcError;
use crate::sender::{Sender, Subscriber};
use ndarray::{s, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{Eig, Inverse, Scalar};

#[derive(Clone, Copy, Debug)]
pub struct CCCIndexValue {
    pub val: f64,
}
#[derive(Default)]
pub struct Index;

impl Index {
    pub fn compute(&self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<f64, CalcError> {
        let n = x.nrows();
        let p = x.ncols();
        let q = *y.iter().max().ok_or("Cant find max")? as usize + 1;
        let xtx = x.t().dot(x);
        let m = &xtx / (x.len() as f64 - 1.);
        let (eigvals_m, _) = m.eig().map_err(|v| v.to_string())?;
        let s = eigvals_m.map(|v| v.re()).sqrt();

        let p_star = s
            .iter()
            .enumerate()
            .filter(|(_, v)| **v >= 1. && **v < q as f64)
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(v, _)| v)
            .ok_or("Cant find p star")?;
        let v_star = s.slice(s![0..p_star]).product();
        let c = (v_star / q as f64).powf(1. / p_star as f64);
        let u = s / c;

        let temp = {
            let a = (0..p_star).map(|i| 1. / (n as f64 + u[i])).sum::<f64>();
            let b = (p_star + 1..p)
                .map(|i| u[i].powi(2) / (n as f64 + u[i]))
                .sum::<f64>();
            let c = (0..p).map(|i| u[i].powi(2)).sum::<f64>();
            (a + b) / c
        };
        let er_squared = 1. - temp * ((n - q).pow(2) / n) as f64 * (1 + 4 / n) as f64;
        let mut z: Array2<f64> = Array2::zeros((n, q));
        for (i, c) in y.iter().enumerate() {
            *z.get_mut((i, *c as usize)).ok_or("Cant get value")? = 1.;
        }

        let r_squared = {
            let ztz = z.t().dot(&z);
            let ztz_inv = Inverse::inv(&ztz).map_err(|e| e.to_string())?;
            let x_ = ztz_inv.dot(&z.t()).dot(x);

            1. - (&xtx - x_.t().dot(&z.t()).dot(&z).dot(&x_)).diag().sum() / xtx.diag().sum()
        };
        let ccc_p1 = ((1. - er_squared) / (1. - r_squared)).ln();
        let ccc_p2 = ((n * p_star) as f64 / 2.).sqrt() / (0.001 + er_squared).powf(1.2);
        let value = ccc_p1 - ccc_p2;

        Ok(value)
    }
}

pub struct Node<'a> {
    index: Index,
    sender: Sender<'a, CCCIndexValue>,
}

impl<'a> Node<'a> {
    pub fn new(sender: Sender<'a, CCCIndexValue>) -> Self {
        Self {
            index: Index,
            sender,
        }
    }
}

impl<'a> Subscriber<(ArrayView2<'a, f64>, ArrayView1<'a, i32>)> for Node<'a> {
    fn recieve_data(
        &mut self,
        data: Result<(ArrayView2<'a, f64>, ArrayView1<'a, i32>), CalcError>,
    ) {
        let res = match data.as_ref() {
            Ok((x, y)) => self.index.compute(x, y).map(|val| CCCIndexValue { val }),
            Err(err) => Err(err.clone()),
        };
        self.sender.send_to_subscribers(res);
    }
}
