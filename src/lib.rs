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

    use std::{
        collections::HashMap,
        ops::Sub,
        sync::{Arc, Mutex},
    };

    use ndarray::{
        parallel::prelude::{
            IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
            ParallelIterator,
        },
        ArcArray2, Array1, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, NdProducer,
        ViewRepr,
    };
    use numpy::{
        ndarray::{Array2, Zip},
        npyffi::npy_int32,
    };
    use pyo3::{exceptions::PyValueError, prelude::*};

    use super::*;

    struct Grouper {
        pub data: HashMap<i32, Array2<f64>>,
    }
    // impl<T> From<ArrayViewD<'_, T>> for Grouper<T> {
    //     fn from(array: ArrayViewD<'_, T>) -> Self {
    //         Self::group(array)
    //     }
    // }
    impl Grouper {
        fn group(x: ArrayView2<'_, f64>, y: ArrayView1<i32>) -> Self {
            let clusters: Mutex<HashMap<i32, Array2<f64>>> = Mutex::new(HashMap::new());
            //let mut clusters: Mutex<Vec<Array2<f64>>> = Mutex::new(Vec::default());
            Zip::from(x.rows()).and(y).par_for_each(|x, y| {
                let mut clusters = clusters.lock().unwrap();
                if !clusters.contains_key(y) {
                    clusters.insert(*y, Array2::default((0, x.shape()[1])));
                }
                let group = clusters.get_mut(y).unwrap();
                group.push_row(x).unwrap();
            });
            Self {
                data: clusters.into_inner().unwrap(),
            }
        }
    }

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

    fn calc_clusters_centers(grouper: &Grouper) -> HashMap<i32, Array1<f64>> {
        let mut retval: HashMap<i32, Array1<f64>> = HashMap::new();
        for (index, val) in grouper.data.iter() {
            retval.insert(*index, val.sum_axis(Axis(0)));
        }
        retval
    }

    fn find_average_euclidean_distance_for_point(
        point: &ArrayView1<f64>,
        cluster: &ArrayView2<f64>,
    ) -> f64 {
        let mut store = Vec::default();
        cluster
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| point.sub(&row))
            .collect_into_vec(&mut store);

        let store: Vec<f64> = store
            .into_par_iter()
            .map(|item| {
                let res = item.dot(&item);
                f64::sqrt(res)
            })
            .collect();
        let n = store.len() as f64;
        let res: f64 = store.into_par_iter().sum();
        res / n
    }

    fn silhouette_index_calc(x: ArrayViewD<f64>, y: ArrayViewD<i32>) -> Result<f64, String> {
        Ok(0.8)
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("silhouette_score", m.getattr("silhouette_score")?)?;
        Ok(())
    }
}
