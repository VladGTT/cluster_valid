use pyo3::prelude::*;
#[pymodule]
mod rust_ext {

    use std::{
        collections::{HashMap, HashSet},
        usize,
    };

    use ndarray::{
        linalg::Dot,
        parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
        Array1, ArrayView1, ArrayView2, Axis,
    };
    use numpy::{
        ndarray::{Array2, Zip},
        npyffi::npy_int32,
        PyReadonlyArrayDyn,
    };
    use pyo3::{exceptions::PyValueError, prelude::*};

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("silhouette_score", m.getattr("silhouette_score")?)?;
        m.add("davies_bouldin_score", m.getattr("davies_bouldin_score")?)?;
        m.add(
            "calinski_harabasz_score",
            m.getattr("calinski_harabasz_score")?,
        )?;
        Ok(())
    }

    //
    //
    //
    //
    //
    //
    //
    //

    fn group(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<i32>,
    ) -> Result<HashMap<i32, Array2<f64>>, String> {
        let clusters = y.into_par_iter().map(|i| *i).collect::<HashSet<i32>>();

        clusters
            .into_par_iter()
            .map(|iter| {
                let vec = Zip::from(y)
                    .and(x.rows())
                    .into_par_iter()
                    .filter(|(i, _)| **i == iter)
                    .map(|(_, val)| val.to_owned())
                    .collect::<Vec<Array1<f64>>>();

                let shape = (vec.len(), vec[0].dim());
                let vec: Vec<f64> = vec.par_iter().flatten().map(|i| *i).collect();
                let matrix: Array2<f64> = Array2::from_shape_vec(shape, vec)
                    .map_err(|_| "Cant create cluster".to_string())?;

                Ok((iter, matrix))
            })
            .collect::<Result<HashMap<i32, Array2<f64>>, String>>()
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

    //
    //
    //
    //
    //
    //
    //
    //
    //

    #[pyfunction]
    fn silhouette_score<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
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

        let result = silhouette_index_calc(x, y);
        result.map_err(PyValueError::new_err)
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

    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //

    #[pyfunction]
    fn davies_bouldin_score<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
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

        let result = davies_bouldin_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn davies_bouldin_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let groups = group(x, y)?;
        let centers = calc_clusters_centers(&groups);

        let variances = (&groups)
            .into_par_iter()
            .map(|(i, val)| {
                let center = centers.get(i).ok_or("Can't get centroid")?;
                let dists_to_center = val
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|row| find_euclidean_distance(&row, &center.view()))
                    .collect::<Vec<f64>>();

                let mean =
                    (&dists_to_center).into_par_iter().sum::<f64>() / dists_to_center.len() as f64;
                Ok((*i, mean))
            })
            .collect::<Result<HashMap<i32, f64>, String>>()?;

        let mut temp: Vec<f64> = Vec::default();
        let n_clusters = groups.keys().len();
        for i in 0..n_clusters {
            for j in 0..n_clusters {
                if i != j {
                    let value = (variances[&(i as i32)] + variances[&(j as i32)])
                        / find_euclidean_distance(
                            &centers[&(i as i32)].view(),
                            &centers[&(j as i32)].view(),
                        );
                    temp.push(value);
                }
            }
        }
        let res = temp
            .into_par_iter()
            .max_by(|x, y| x.total_cmp(y))
            .ok_or("Can't find max element".to_string())?
            / n_clusters as f64;

        Ok(res)
    }

    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //

    #[pyfunction]
    fn calinski_harabasz_score<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
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

        let result = calinski_harabasz_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn calinski_harabasz_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let number_of_objects = x.axis_iter(Axis(0)).len() as f64;
        let data_centroid = x.sum_axis(Axis(0)) / number_of_objects;
        let clusters = group(x, y)?;
        let number_of_objects_in_clusters = (&clusters)
            .into_par_iter()
            .map(|(i, val)| (*i, val.axis_iter(Axis(0)).len()))
            .collect::<HashMap<i32, usize>>();
        let clusters_centroids = calc_clusters_centers(&clusters);
        let number_of_clusters = clusters.keys().len() as f64;

        let between_group_dispersion = clusters_centroids
            .iter()
            .zip(number_of_objects_in_clusters)
            .filter(|((i, _), (j, _))| **i == *j)
            .map(|((_, c), (_, n))| {
                let dif = c - (&data_centroid);
                (n as f64) * dif.dot(&dif)
            })
            .sum::<f64>();

        let within_group_dispersion = clusters
            .into_par_iter()
            .map(|(i, val)| {
                let center = &clusters_centroids[&i].view();
                let sum = val
                    .axis_iter(Axis(0))
                    .map(|row| {
                        let dif = &row - center;
                        dif.dot(&dif)
                    })
                    .sum::<f64>();
                sum
            })
            .sum::<f64>();

        let res = (between_group_dispersion / (number_of_clusters - 1.0))
            / (within_group_dispersion / (number_of_objects - number_of_clusters));
        Ok(res)
    }
}
