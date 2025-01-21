pub mod clusters;
pub mod clusters_centroids;
pub mod pairs_and_distances;
pub mod raw_data;

// #[derive(Default, Debug)]
// pub struct ResultReader<T> {
//     res: Option<Arc<Result<T, CalcError>>>,
// }
// impl<T: Clone> ResultReader<T> {
//     pub fn get(&self) -> Option<Result<T, CalcError>> {
//         self.res.as_ref().map(|d| d.deref().clone())
//     }
// }
// impl<T: Clone> Subscriber<T> for ResultReader<T> {
//     fn recieve_data(&mut self, data: std::sync::Arc<Result<T, CalcError>>) {
//         self.res = Some(data);
//     }
// }
// pub fn find_euclidean_distance(point1: &ArrayView1<f64>, point2: &ArrayView1<f64>) -> f64 {
//     calc_vector_euclidean_length(&(point2 - point1).view())
// }
// pub fn calc_vector_euclidean_length(vec: &ArrayView1<f64>) -> f64 {
//     f64::sqrt(vec.dot(vec))
// }
//
// pub fn calc_matrix_determinant(matrix: &ArrayView2<f64>) -> Result<f64, String> {
//     if matrix.nrows() != matrix.ncols() {
//         return Err("Matrix is not square".to_string());
//     }
//     let mut matrix_U: Array2<f64> = matrix.to_owned();
//
//     // Gaussian elemination
//     for i in 0..matrix_U.nrows() {
//         for j in i + 1..matrix_U.nrows() {
//             let koef = matrix_U[[j, i]] / matrix_U[[i, i]];
//             let temp = matrix_U.row(i).mul(koef);
//             matrix_U.row_mut(j).sub_assign(&temp);
//         }
//     }
//
//     let retval = matrix_U.diag().product();
//     Ok(retval)
// }
