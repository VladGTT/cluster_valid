use pyo3::prelude::*;

use ndarray::{
    parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    Array1, Array2, ArrayView1, ArrayView2,
};
use numpy::{npyffi::npy_int32, PyReadonlyArrayDyn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

mod calc_error;
mod helpers;
mod indexes;
#[cfg(test)]
mod tests;
use helpers::{
    clusters::ClustersNode, clusters_centroids::ClustersCentroidsNode, raw_data::RawDataNode,
};

use indexes::{ball_hall, Sender};
#[pymodule]
mod rust_ext {
    //
    use super::*;
    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // m.add("silhouette_score", m.getattr("silhouette_score")?)?;
        // m.add("davies_bouldin_score", m.getattr("davies_bouldin_score")?)?;
        // m.add(
        //     "calinski_harabasz_score",
        //     m.getattr("calinski_harabasz_score")?,
        // )?;
        // m.add("c_index", m.getattr("c_index")?)?;
        // m.add("gamma_index", m.getattr("gamma_index")?)?;
        m.add("ball_hall_index", m.getattr("ball_hall_index")?)?;
        // m.add("tau_index", m.getattr("tau_index")?)?;
        // m.add("dunn_index", m.getattr("dunn_index")?)?;
        // m.add("sd_index", m.getattr("sd_index")?)?;
        // m.add("tracew_index", m.getattr("tracew_index")?)?;
        // m.add("trcovw_index", m.getattr("trcovw_index")?)?;
        // m.add("ratkowsky_index", m.getattr("ratkowsky_index")?)?;
        // m.add("mcclain_index", m.getattr("mcclain_index")?)?;
        // m.add("gplus_index", m.getattr("gplus_index")?)?;
        // m.add("ptbiserial_index", m.getattr("ptbiserial_index")?)?;
        Ok(())
    }
    //     macro_rules! wrapper {
    //         ($func:expr,$matrix:expr,$clusters:expr) => {{
    //             let x = $matrix.as_array();
    //             let y = $clusters.as_array();
    //
    //             let shape = match (x.shape().first(), x.shape().get(1)) {
    //                 (Some(val_x), Some(val_y)) => (*val_x, *val_y),
    //                 _ => return Err(PyValueError::new_err("x is not 2 dimentional".to_string())),
    //             };
    //
    //             let x = x
    //                 .into_shape(shape)
    //                 .map_err(|msg| PyValueError::new_err(format!("{msg}")))?;
    //
    //             let y = y
    //                 .into_shape(shape.0)
    //                 .map_err(|msg| PyValueError::new_err(format!("{msg}")))?;
    //
    //             let result = $func(x, y);
    //             result.map_err(PyValueError::new_err)
    //         }};
    //     }
    //
    //     #[pyfunction]
    //     pub fn silhouette_score<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::silhouette_index, x, y)
    //     }
    //     #[pyfunction]
    //     pub fn davies_bouldin_score<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::davies_bouldin_index, x, y)
    //     }
    //     #[pyfunction]
    //     pub fn calinski_harabasz_score<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::calinski_harabasz_index, x, y)
    //     }
    //     #[pyfunction]
    //     pub fn c_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::c_index, x, y)
    //     }
    //     #[pyfunction]
    //     pub fn gamma_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::gamma_index, x, y)
    //     }

    #[pyfunction]
    pub fn ball_hall_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) {
        let matrix = x.as_array().into_shape((2, 1)).unwrap();
        let cls = y.as_array().into_shape(1).unwrap();

        let index = Mutex::new(ball_hall::Node::default());
        let clusters_centroids = Mutex::new(ClustersCentroidsNode::with_sender(Sender::new(vec![
            &index,
        ])));
        let clusters = Mutex::new(ClustersNode::with_sender(Sender::new(vec![
            &index,
            &clusters_centroids,
        ])));
        let mut raw_data = RawDataNode::new(
            (&matrix, &cls),
            Sender::new(vec![&index, &clusters, &clusters_centroids]),
        );
        raw_data.compute();
    }
    //
    // clusters_centroids
    //     .lock()
    //     .unwrap()
    //     .subscribee
    //     .add_subscriber(&index);
    // }
    //     #[pyfunction]
    //     pub fn tau_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::tau_index, x, y)
    //     }
    //     #[pyfunction]
    //     pub fn dunn_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::dunn_index, x, y)
    //     }
    //     #[pyfunction]
    //     pub fn sd_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<(f64, f64)> {
    //         wrapper!(indexes::sd_index, x, y)
    //     }
    //
    //     #[pyfunction]
    //     pub fn tracew_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::tracew_index, x, y)
    //     }
    //
    //     #[pyfunction]
    //     pub fn trcovw_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::trcovw_index, x, y)
    //     }
    //
    //     #[pyfunction]
    //     pub fn ratkowsky_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::ratkowsky_index, x, y)
    //     }
    //     #[pyfunction]
    //     pub fn mcclain_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::mcclain_index_calc, x, y)
    //     }
    //
    //     #[pyfunction]
    //     pub fn gplus_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::gplus_index, x, y)
    //     }
    //     #[pyfunction]
    //     pub fn ptbiserial_index<'py>(
    //         x: PyReadonlyArrayDyn<'py, f64>,
    //         y: PyReadonlyArrayDyn<'py, npy_int32>,
    //     ) -> PyResult<f64> {
    //         wrapper!(indexes::ptbiserial_index, x, y)
    //     }
}
