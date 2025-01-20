mod calc_error;
mod helpers;
mod index_tree;
mod indexes;
#[cfg(test)]
mod tests;

use pyo3::prelude::*;
#[pymodule]
mod rust_ext {
    use super::*;
    use core::f64;
    use index_tree::{IndexTree, IndexTreeReturnValue};
    use numpy::{npyffi::npy_int32, PyReadonlyArrayDyn};
    use pyo3::exceptions::PyValueError;
    #[pyclass(frozen)]
    #[derive(Default, Debug)]
    pub struct IndexTreeConfig {
        pub ball_hall: bool,
        pub davies_bouldin: bool,
        pub c_index: bool,
        pub calinski_harabasz: bool,
        pub dunn: bool,
        pub silhoutte: bool,
    }
    #[pymethods]
    impl IndexTreeConfig {
        #[new]
        fn new(
            ball_hall: bool,
            davies_bouldin: bool,
            c_index: bool,
            calinski_harabasz: bool,
            dunn: bool,
            silhoutte: bool,
        ) -> Self {
            Self {
                ball_hall,
                davies_bouldin,
                c_index,
                dunn,
                calinski_harabasz,
                silhoutte,
            }
        }
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add(
            "compute_indexes_with_config",
            m.getattr("compute_indexes_with_config")?,
        )?;

        Ok(())
    }
    #[pyfunction]
    fn compute_indexes_with_config<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
        config: Py<IndexTreeConfig>,
    ) -> PyResult<Py<IndexTreeReturnValue>> {
        let py = x.py();
        let x = x.as_array();
        let y = y.as_array();
        let shape = match (x.shape().first(), x.shape().get(1)) {
            (Some(val_x), Some(val_y)) => (*val_x, *val_y),
            _ => return Err(PyValueError::new_err("x is not 2 dimentional".to_string())),
        };
        let x = x
            .into_shape(shape)
            .map_err(|msg| PyValueError::new_err(format!("{msg}")))?;
        let y = y
            .into_shape(shape.0)
            .map_err(|msg| PyValueError::new_err(format!("{msg}")))?;

        //
        let config = config.get();
        let tree = IndexTree::new(config);
        let res = tree.compute((&x, &y));

        Py::new(py, res)
    }
}
