use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::IntoPy;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

// #[pymodule]
// fn rust_ext<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
//     fn silhouette_index_calc(x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> f64 {
//         0.8
//     }
//
//     #[pyfn(m)]
//     #[pyo3(name = "silhouette_score")]
//     fn wrapper<'py>(
//         x: PyReadonlyArrayDyn<'py, f64>,
//         y: PyReadonlyArrayDyn<'py, f64>,
//     ) -> Bound<'py, f64> {
//         let x = x.as_array();
//         let y = y.as_array();
//         silhouette_index_calc(x, y)
//     }
//
//     // fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
//     //     a * &x + &y
//     // }
//     //
//     // #[pyfn(m)]
//     // #[pyo3(name = "axpy")]
//     // fn axpy_py<'py>(
//     //     py: Python<'py>,
//     //     a: f64,
//     //     x: PyReadonlyArrayDyn<'py, f64>,
//     //     y: PyReadonlyArrayDyn<'py, f64>,
//     // ) -> Bound<'py, PyArrayDyn<f64>> {
//     //     let x = x.as_array();
//     //     let y = y.as_array();
//     //     let z = axpy(a, x, y);
//     //     z.into_pyarray_bound(py)
//     // }
//
//     Ok(())
//
// }

#[pymodule]
mod rust_ext {

    use std::sync::Arc;

    use ndarray::{
        parallel::prelude::IntoParallelIterator, ArcArray2, Array1, ArrayView, ArrayView1,
        NdProducer,
    };
    use numpy::{
        ndarray::{Array2, Zip},
        npyffi::npy_int32,
    };
    use pyo3::{exceptions::PyValueError, prelude::*};

    use super::*;

    // fn calc_cohersion<'py>(
    //     i: i32,
    //     x: PyReadonlyArrayDyn<'py, f64>,
    //     y: PyReadonlyArrayDyn<'py, npy_int32>,
    // ) -> f64 {
    //
    //
    //
    //
    // }

    #[pyfunction]
    fn silhouette_score<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        let x = x.as_array();
        let y = y.as_array();

        let result = silhouette_index_calc(x, y);
        result.map_err(|msg| PyValueError::new_err(msg))
    }

    fn calc_clusters_centers(x: ArrayViewD<f64>, y: ArrayViewD<i32>) -> Arc<Array2<f64>> {
        let mut clusters = Arc::new(Array2::<f64>::zeros((1, x.shape()[1])));
        let num_cluster_elements = Array1::<i32>::zeros(1);
        Zip::from(x.rows()).and(y).par_for_each(|x, y| {
            while *y as usize - clusters.shape()[0] >= 0 {
                let new_row = Array1::<f64>::zeros(x.shape()[1]);
                *clusters.push_row(ArrayView1::<f64>::from(&new_row));
            }
        });

        clusters
    }
    fn silhouette_index_calc(x: ArrayViewD<f64>, y: ArrayViewD<i32>) -> Result<f64, String> {
        // Finding center of clusters
        // for row in Zip::from(x.rows).and(y) {
        //
        // }

        // let mut temp = Array3::<f64>::zeros((1, x_shape[1], 1));
        // let zipped = Zip::from(x.rows()).and(y);
        // zipped.par_for_each(|x, y| {});
        Ok(0.8)
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("silhouette_score", m.getattr("silhouette_score")?)?;
        Ok(())
    }
}
