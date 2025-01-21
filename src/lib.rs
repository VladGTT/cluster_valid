mod calc_error;
mod helpers;
mod index_tree;
mod indexes;
mod sender;
#[cfg(test)]
mod tests;

use pyo3::prelude::*;
#[pymodule]
mod rust_ext {
    use super::*;
    use core::f64;
    use index_tree::{IndexTreeBuilder, IndexTreeReturnValue};
    use numpy::{npyffi::npy_int32, PyReadonlyArray1, PyReadonlyArray2};

    #[pyclass(frozen)]
    #[derive(Default, Debug)]
    struct IndexTreeConfig {
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
        m.add("compute_indexes", m.getattr("compute_indexes")?)?;
        m.add("Config", m.getattr("IndexTreeConfig")?)?;
        Ok(())
    }
    #[pyfunction]
    fn compute_indexes<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, npy_int32>,
        config: Py<IndexTreeConfig>,
    ) -> PyResult<Py<IndexTreeReturnValue>> {
        let x = x.as_array();
        let y = y.as_array();

        let tree = {
            let config = config.get();
            let mut builder = IndexTreeBuilder::default();
            if config.ball_hall {
                builder = builder.add_ball_hall();
            }
            builder.finish()
        };
        Py::new(py, tree.compute((x, y)))
    }
}
