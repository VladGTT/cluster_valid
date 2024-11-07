use std::ops::{AddAssign, Mul, SubAssign};

use super::*;
use itertools::Itertools;

pub fn calc_clusters(y: &ArrayView1<i32>) -> HashMap<i32, Array1<usize>> {
    let mut cluster_indexes_with_counter = y
        .iter()
        .counts()
        .into_iter()
        .map(|(i, n)| (*i, (Array1::zeros(n), 0)))
        .collect::<HashMap<i32, (Array1<usize>, usize)>>();
    for (i, c) in y.iter().enumerate() {
        let (arr, ctr) = cluster_indexes_with_counter.get_mut(c).unwrap();
        arr[*ctr] = i;
        ctr.add_assign(1);
    }

    cluster_indexes_with_counter
        .into_iter()
        .map(|(i, (arr, _))| (i, arr))
        .collect::<HashMap<i32, Array1<usize>>>()
}
pub fn calc_clusters_centers(
    clusters_indexes: &HashMap<i32, Array1<usize>>,
    data: &ArrayView2<f64>,
) -> HashMap<i32, Array1<f64>> {
    clusters_indexes
        .into_par_iter()
        .map(|(c, arr)| {
            let mut sum: Array1<f64> = Array1::zeros(data.ncols());
            for i in arr {
                sum += &data.row(*i);
            }
            (*c, sum / arr.len() as f64)
        })
        .collect::<HashMap<i32, Array1<f64>>>()
}
pub fn find_euclidean_distance(point1: &ArrayView1<f64>, point2: &ArrayView1<f64>) -> f64 {
    calc_vector_euclidean_length(&(point2 - point1).view())
}
pub fn calc_vector_euclidean_length(vec: &ArrayView1<f64>) -> f64 {
    f64::sqrt(vec.dot(vec))
}

pub fn calc_matrix_determinant(matrix: &ArrayView2<f64>) -> Result<f64, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err("Matrix is not square".to_string());
    }
    let mut matrix_U: Array2<f64> = matrix.to_owned();

    // Gaussian elemination
    for i in 0..matrix_U.nrows() {
        for j in i + 1..matrix_U.nrows() {
            let koef = matrix_U[[j, i]] / matrix_U[[i, i]];
            let temp = matrix_U.row(i).mul(koef);
            matrix_U.row_mut(j).sub_assign(&temp);
        }
    }

    let retval = matrix_U.diag().product();
    Ok(retval)
}
