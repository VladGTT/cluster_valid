use pyo3::prelude::*;
#[pymodule]
mod rust_ext {

    use std::{collections::HashMap, sync::Mutex};

    use ndarray::{
        parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
        Array1, ArrayView1, ArrayView2, Axis,
    };
    use numpy::{
        ndarray::{Array2, Zip},
        npyffi::npy_int32,
        PyReadonlyArrayDyn,
    };
    use pyo3::{exceptions::PyValueError, prelude::*};

    #[pyfunction]
    fn silhouette_score<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        let x = x.as_array();
        let y = y.as_array();

        let shape = match (x.shape().get(0), x.shape().get(1)) {
            (Some(val_x), Some(val_y)) => (*val_x, *val_y),
            _ => return Err(PyValueError::new_err("x is not 2 dimentional".to_string())),
        };

        let x = x
            .into_shape(shape)
            .map_err(|msg| PyValueError::new_err(format!("{msg}")))?;

        let y = y
            .into_shape(shape.0)
            .map_err(|msg| PyValueError::new_err(format!("{msg}")))?;

        let result = silhouette_index_calc(x, y);
        result.map_err(|msg| PyValueError::new_err(msg))
    }

    fn group(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<i32>,
    ) -> Result<HashMap<i32, Array2<f64>>, String> {
        let clusters: Mutex<HashMap<i32, Array2<f64>>> = Mutex::new(HashMap::new());
        let _ = Zip::from(x.rows())
            .and(y)
            .into_par_iter()
            .map(|(x, y)| {
                let mut clusters = clusters
                    .lock()
                    .map_err(|_| "Cant get a lock on hashmap".to_string())?;
                if !clusters.contains_key(y) {
                    clusters.insert(*y, Array2::default((0, x.shape()[0])));
                }
                let group = clusters
                    .get_mut(y)
                    .ok_or("Can`t get group from hashmap".to_string())?;
                group
                    .push(Axis(0), x)
                    .map_err(|_| "Couldn`t add point to cluster".to_string())?;
                Ok(())
            })
            .collect::<Result<(), String>>()?;
        let res = clusters.into_inner().map_err(|msg| format!("{msg}"))?;
        Ok(res)
    }

    fn calc_clusters_centers(groups: &HashMap<i32, Array2<f64>>) -> HashMap<i32, Array1<f64>> {
        let retval = groups
            .into_par_iter()
            .map(|(i, val)| {
                let center = val.sum_axis(Axis(0));
                let len = val.shape()[0] as f64;
                (*i, center / len)
            })
            .collect::<HashMap<i32, Array1<f64>>>();
        retval
    }

    fn find_euclidean_distance(point1: &ArrayView1<f64>, point2: &ArrayView1<f64>) -> f64 {
        let sub_res = point2 - point1;
        f64::sqrt(sub_res.dot(&sub_res))
    }

    fn silhouette_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let groups = group(x, y)?;
        let centers = calc_clusters_centers(&groups);
        let scores = Zip::from(x.rows())
            .and(y)
            .into_par_iter()
            .map(|(x, y)| {
                let min = (&centers)
                    .into_par_iter()
                    .filter(|val| *val.0 != *y)
                    .map(|(i, row)| (i, find_euclidean_distance(&x, &row.view())))
                    .min_by(|(_, x), (_, y)| x.total_cmp(y))
                    .ok_or("Cant find closest cluster".to_string())?;

                let nearest_cluster = groups
                    .get(min.0)
                    .ok_or("Cant get a lock on nearest cluster")?;
                let nearest_cluster_distances = nearest_cluster
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|row| find_euclidean_distance(&x, &row))
                    .collect::<Vec<f64>>();

                let a: f64 = nearest_cluster_distances.par_iter().sum::<f64>()
                    / nearest_cluster_distances.len() as f64;

                let point_cluster = groups
                    .get(y)
                    .ok_or("Cant get a lock on point own cluster")?;
                let point_cluster_distances = point_cluster
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|row| find_euclidean_distance(&x, &row))
                    .collect::<Vec<f64>>();

                let b: f64 = point_cluster_distances.par_iter().sum::<f64>()
                    / (point_cluster_distances.len() - 1) as f64;

                Ok((a - b) / f64::max(a, b))
            })
            .collect::<Result<Vec<f64>, String>>()?;

        let res = scores.par_iter().sum::<f64>() / scores.len() as f64;
        Ok(res)
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("silhouette_score", m.getattr("silhouette_score")?)?;
        Ok(())
    }
}
