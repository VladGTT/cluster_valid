mod calc_error;
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
    use indexes::ptbiserial;
    use numpy::{npyffi::npy_int32, PyReadonlyArray1, PyReadonlyArray2};

    #[pyclass(frozen)]
    #[derive(Default, Debug)]
    struct IndexTreeConfig {
        pub ball_hall: bool,
        pub davies_bouldin: bool,
        pub c_index: bool,
        pub calinski_harabasz: bool,
        pub dunn: bool,
        pub silhouette: bool,
        pub rubin: bool,
        pub mariott: bool,
        pub scott: bool,
        pub friedman: bool,
        pub tau: bool,
        pub gamma: bool,
        pub gplus: bool,
        pub tracew: bool,
        pub mcclain: bool,
        pub ptbiserial: bool,
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
            silhouette: bool,
            rubin: bool,
            mariott: bool,
            scott: bool,
            friedman: bool,
            tau: bool,
            gamma: bool,
            gplus: bool,
            tracew: bool,
            mcclain: bool,
            ptbiserial: bool,
        ) -> Self {
            Self {
                ball_hall,
                davies_bouldin,
                c_index,
                dunn,
                calinski_harabasz,
                silhouette,
                rubin,
                mariott,
                scott,
                friedman,
                tau,
                gamma,
                gplus,
                tracew,
                mcclain,
                ptbiserial,
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
            if config.davies_bouldin {
                builder = builder.add_davies_bouldin();
            }
            if config.silhouette {
                builder = builder.add_silhouette();
            }
            if config.c_index {
                builder = builder.add_c_index();
            }
            if config.dunn {
                builder = builder.add_dunn();
            }
            if config.calinski_harabasz {
                builder = builder.add_calinski_harabasz();
            }
            if config.rubin {
                builder = builder.add_rubin();
            }
            if config.mariott {
                builder = builder.add_mariott();
            }
            if config.scott {
                builder = builder.add_scott();
            }
            if config.friedman {
                builder = builder.add_friedman();
            }
            if config.tau {
                builder = builder.add_tau();
            }
            if config.gamma {
                builder = builder.add_gamma();
            }
            if config.gplus {
                builder = builder.add_gplus();
            }
            if config.tracew {
                builder = builder.add_tracew();
            }
            if config.mcclain {
                builder = builder.add_mcclain();
            }
            if config.ptbiserial {
                builder = builder.add_ptbiserial();
            }
            builder.finish()
        };
        Py::new(py, tree.compute((x, y)))
    }
}
