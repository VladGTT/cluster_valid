use pyo3::prelude::*;
#[pymodule]
mod rust_ext {

    use ndarray::{
        parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
        Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
    };
    use numpy::{npyffi::npy_int32, PyReadonlyArrayDyn};
    use pyo3::{exceptions::PyValueError, prelude::*};
    use std::{
        collections::{HashMap, HashSet},
        iter::zip,
        ops::AddAssign,
    };

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("silhouette_score", m.getattr("silhouette_score")?)?;
        m.add("davies_bouldin_score", m.getattr("davies_bouldin_score")?)?;
        m.add(
            "calinski_harabasz_score",
            m.getattr("calinski_harabasz_score")?,
        )?;
        m.add("c_index", m.getattr("c_index")?)?;
        m.add("gamma_index", m.getattr("gamma_index")?)?;
        m.add("ball_hall_index", m.getattr("ball_hall_index")?)?;
        m.add("tau_index", m.getattr("tau_index")?)?;
        m.add("dunn_index", m.getattr("dunn_index")?)?;
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

        let stds = (&groups)
            .into_par_iter()
            .map(|(i, val)| {
                let center = centers[i].view();
                let dists_to_center = val
                    .axis_iter(Axis(0))
                    .map(|row| find_euclidean_distance(&row, &center))
                    .collect::<Vec<f64>>();

                // println!("Dists to center {dists_to_center:?}");

                let mean = dists_to_center.iter().sum::<f64>() / dists_to_center.len() as f64;

                // println!(
                //grouping data into clusters
                //let clusters = group(x, y);

                //     "Center {i}: {} Mean: {} len: {}\n\n",
                //     center,
                //     mean,
                //     dists_to_center.len() as f64
                // );

                Ok((*i, mean))
            })
            .collect::<Result<HashMap<i32, f64>, String>>()?;

        // println!("stds: {:?}", stds);

        let mut temp: Vec<f64> = Vec::default();
        let n_clusters = groups.keys().len();

        // println!("n_clusters: {}", n_clusters);

        for i in 0..n_clusters as i32 {
            for j in 0..n_clusters as i32 {
                if i != j {
                    let value = (stds[&i] + stds[&j])
                        / find_euclidean_distance(&centers[&i].view(), &centers[&j].view());
                    temp.push(value);
                }
            }
        }

        // println!("temp: {:?}", temp);

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

        let between_group_dispersion = (&clusters_centroids)
            .into_par_iter()
            .map(|(i, c)| {
                let dif = c - &data_centroid;
                let n = number_of_objects_in_clusters[&i];

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
    fn c_index<'py>(
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

        let result = c_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn c_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        //transforming dataset into hashmap
        let clusters = group(x, y)?;

        //calculating Nw  -- total number of pairs of observations belonging to the same cluster
        let number_of_pairs_in_clusters = (&clusters)
            .into_par_iter()
            .map(|(_, val)| {
                let n = val.axis_iter(Axis(0)).len() as f64;
                n * (n - 1.0) / 2.0
            })
            .sum::<f64>() as usize;

        //calculating distances beetween all possible pars of points in dataset
        let mut distances: Vec<f64> = Vec::default();

        for (i, row1) in x.axis_iter(Axis(0)).enumerate() {
            for (j, row2) in x.axis_iter(Axis(0)).enumerate() {
                if i < j {
                    distances.push(find_euclidean_distance(&row1, &row2));
                }
            }
        }

        //sorting by min
        distances.sort_unstable_by(|a, b| a.total_cmp(b));

        //calculating sum of Nw minimum and maximum distances
        let mut sum_of_minimum_distances = 0.0;
        let mut sum_of_maximum_distances = 0.0;
        for i in 0..number_of_pairs_in_clusters {
            sum_of_minimum_distances += distances[i];
            sum_of_maximum_distances += distances[(distances.len() - 1) - i];
        }

        //calculating Sw -- sum of the within-cluster distances
        let sum_of_withincluster_distances = (&clusters)
            .into_par_iter()
            .map(|(_, val)| {
                let mut distances: Vec<f64> = Vec::default();
                for (i, row1) in val.axis_iter(Axis(0)).enumerate() {
                    for (j, row2) in val.axis_iter(Axis(0)).enumerate() {
                        if i < j {
                            distances.push(find_euclidean_distance(&row1, &row2));
                        }
                    }
                }
                distances.iter().sum::<f64>()
            })
            .sum::<f64>();

        //calculating c_index value
        Ok((sum_of_withincluster_distances - sum_of_minimum_distances)
            / (sum_of_maximum_distances - sum_of_minimum_distances))
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

    #[pyfunction]
    fn gamma_index<'py>(
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

        let result = gamma_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn gamma_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let mut distances: Vec<f64> = Vec::default();
        let mut pairs_in_the_same_cluster: Vec<i8> = Vec::default();

        //calculating distances beetween pair of points and does they belong to the same cluster
        for (i, (row1, cluster1)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
            for (j, (row2, cluster2)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
                if i < j {
                    pairs_in_the_same_cluster.push((cluster1 != cluster2) as i8); // the same cluster =producer 0, different = 1
                    distances.push(find_euclidean_distance(&row1, &row2));
                }
            }
        }
        let (mut s_plus, mut s_minus) = (0.0, 0.0);

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        //belonging to the same cluster

        for (i, (d1, b1)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
                if i < j && (*b1 == 0 && *b2 == 1) {
                    if d1 < d2 {
                        s_plus += 1.0;
                    }
                    if d1 > d2 {
                        s_minus += 1.0;
                    }
                }
            }
        }
        Ok((s_plus - s_minus) / (s_plus + s_minus))
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

    #[pyfunction]
    fn ball_hall_index<'py>(
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

        let result = ball_hall_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn ball_hall_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let clusters = group(x, y)?;
        let number_of_cluster = clusters.keys().len();
        let clusters_centroids = calc_clusters_centers(&clusters);
        let std = clusters
            .into_par_iter()
            .map(|(i, val)| {
                let center = &clusters_centroids[&i].view();
                val.axis_iter(Axis(0))
                    .map(|row| {
                        let dif = &row - center;
                        dif.dot(&dif)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        Ok(std / number_of_cluster as f64)
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

    #[pyfunction]
    fn tau_index<'py>(
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

        let result = tau_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn tau_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let mut distances: Vec<f64> = Vec::default();
        let mut pairs_in_the_same_cluster: Vec<i8> = Vec::default();

        //calculating distances beetween pair of points and does they belong to the same cluster
        for (i, (row1, cluster1)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
            for (j, (row2, cluster2)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
                if i < j {
                    pairs_in_the_same_cluster.push((cluster1 != cluster2) as i8); // the same cluster = 0, different = 1
                    distances.push(find_euclidean_distance(&row1, &row2));
                }
            }
        }

        let number_of_pairs_in_the_same_cluster = pairs_in_the_same_cluster
            .iter()
            .filter(|i| **i == 0)
            .count();

        let total_number_of_pairs = x.len() * (x.len() - 1) / 2;

        let number_of_pairs_in_different_clusters =
            total_number_of_pairs - number_of_pairs_in_the_same_cluster;

        let (mut s_plus, mut s_minus) = (0.0, 0.0);

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        //belonging to the same cluster

        let mut ties: HashMap<u64, i32> = HashMap::default();

        for (i, (d1, b1)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
                if i < j && (*b1 == 0 && *b2 == 1) {
                    if d1 < d2 {
                        s_plus += 1.0;
                    }
                    if d1 > d2 {
                        s_minus += 1.0;
                    }
                    if d1 == d2 {
                        ties.entry(d1.to_bits())
                            .and_modify(|val| *val += 1)
                            .or_insert(0);
                    }
                }
            }
        }

        let v0 = (total_number_of_pairs * (total_number_of_pairs - 1)) as f64 / 2.0;
        let v1 = ties
            .iter()
            .map(|(_, c)| (c * (c - 1)) as f64 / 2.0)
            .sum::<f64>();
        let v2 = (number_of_pairs_in_different_clusters
            * (number_of_pairs_in_different_clusters - 1)) as f64
            / 2.0
            + (number_of_pairs_in_the_same_cluster * (number_of_pairs_in_the_same_cluster - 1))
                as f64
                / 2.0;

        Ok((s_plus - s_minus) / f64::sqrt((v0 - v1) * (v0 - v2)))
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

    #[pyfunction]
    fn dunn_index<'py>(
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

        let result = dunn_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn dunn_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let mut intercluster_distances: Vec<f64> = Vec::default();
        let mut intracluster_distances: Vec<f64> = Vec::default();

        for (i, (row1, cluster1)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
            for (j, (row2, cluster2)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
                if i < j {
                    if cluster1 == cluster2 {
                        intracluster_distances.push(find_euclidean_distance(&row1, &row2));
                    } else {
                        intercluster_distances.push(find_euclidean_distance(&row1, &row2));
                    }
                }
            }
        }

        let max_intracluster = intracluster_distances
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .ok_or("Can't find max intracluster distance")?;
        let min_intercluster = intercluster_distances
            .iter()
            .min_by(|x, y| x.total_cmp(y))
            .ok_or("Can't find min intercluster distance")?;

        Ok(min_intercluster / max_intracluster)
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

    #[pyfunction]
    fn sd_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<(f64, f64)> {
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

        let result = sd_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn sd_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<(f64, f64), String> {
        let features_variances = x.var_axis(Axis(1), 0.);

        let clusters = group(x, y)?;
        let clusters_centroids = calc_clusters_centers(&clusters);
        let cluster_features_variances = clusters
            .iter()
            .map(|(_, val)| val.var_axis(Axis(1), 0.))
            .collect::<Vec<Array1<f64>>>();

        let S = Array1::from_vec(
            cluster_features_variances
                .iter()
                .map(|i| i.dot(i).sqrt())
                .collect::<Vec<f64>>(),
        )
        .mean()
        .ok_or("Can't calculate mean")?
            / features_variances.dot(&features_variances).sqrt();

        let D = |cluster_centroids: &HashMap<i32, Array1<f64>>| -> f64 {
            let mut d = 0.0;
            let mut d_max = -1.0;
            let mut d_min = f64::MAX;
            for (i, row1) in cluster_centroids {
                let mut dist_acum = 0.0;
                for (j, row2) in cluster_centroids {
                    if i < j {
                        let dist = find_euclidean_distance(&row1.view(), &row2.view());
                        dist_acum += dist;

                        if dist > d_max {
                            d_max = dist;
                        }
                        if dist < d_min {
                            d_min = dist;
                        }
                    }
                }
                if dist_acum != 0.0 {
                    d += 1.0 / dist_acum;
                }
            }
            d * d_max / d_min
        };

        // let max_cluster_labels =
        //     Array1::from_vec((0..x.axis_iter(Axis(0)).len() as i32).collect::<Vec<i32>>());
        // let clusters2 = group(x, max_cluster_labels.view())?;
        Ok((S, D(&clusters_centroids)))
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

    #[pyfunction]
    fn tracew_index<'py>(
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

        let result = tracew_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn tracew_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let cluster_centroids = calc_clusters_centers(&group(x, y)?);
        let (n, d) = x.dim();
        let mut diffs: Array2<f64> = Array2::zeros((n, d));
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            let temp = cluster_centroids[y].to_owned() - x;
            diffs.row_mut(i).assign(&temp);
        }

        let retval = diffs.t().dot(&diffs).diag().sum();
        Ok(retval)
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

    #[pyfunction]
    fn trcovw_index<'py>(
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

        let result = trcovw_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn trcovw_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let cluster_centroids = calc_clusters_centers(&group(x, y)?);
        let (n, d) = x.dim();
        let mut diffs: Array2<f64> = Array2::zeros((n, d));
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            let temp = cluster_centroids[y].to_owned() - x;
            diffs.row_mut(i).assign(&temp);
        }
        let w = diffs.t().dot(&diffs);
        let retval = calc_covariance_matrix(w)?.diag().sum();

        Ok(retval)
    }

    fn calc_covariance_matrix(x: Array2<f64>) -> Result<Array2<f64>, String> {
        let calc_covar = |x: ArrayView1<f64>, y: ArrayView1<f64>| -> Option<f64> {
            let x_mean = x.mean()?;
            let y_mean = y.mean()?;
            let n = x.len() as f64;
            let retval = zip(x, y)
                .map(|(x, y)| (x_mean - x) * (y_mean - y))
                .sum::<f64>()
                / n;
            Some(retval)
        };
        let (_, d) = x.dim();
        let mut retval: Array2<f64> = Array2::zeros((d, d));
        for (i, r1) in x.axis_iter(Axis(1)).enumerate() {
            for (j, r2) in x.axis_iter(Axis(1)).enumerate() {
                if i <= j {
                    retval[[i, j]] = calc_covar(r1, r2).ok_or("Cant calc covariance")?;
                    retval[[j, i]] = retval[[i, j]];
                }
            }
        }
        Ok(retval)
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

    #[pyfunction]
    fn ratkowsky_index<'py>(
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

        let result = ratkowsky_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn ratkowsky_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let x_mean = x
            .mean_axis(Axis(0))
            .ok_or("Cant compute mean for dataset")?;
        let clusters = group(x, y)?;
        let clusters_centroids = calc_clusters_centers(&clusters);

        let (num_of_elems, num_of_vars) = x.dim();

        let mut bgss: Array1<f64> = Array1::zeros(num_of_vars);
        for (_, c) in clusters_centroids {
            bgss = bgss + (c - &x_mean).pow2();
        }
        bgss *= num_of_elems as f64;

        let tss = x.var_axis(Axis(0), 0.) * num_of_elems as f64;

        let s_squared = (bgss / tss).sum() / num_of_vars as f64;
        Ok((s_squared / clusters.keys().len() as f64).sqrt())
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

    #[pyfunction]
    fn mcclain_index<'py>(
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

        let result = mcclain_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn mcclain_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let (mut num_pairs_the_same_clust, mut num_pairs_dif_clust): (f64, f64) = (0., 0.);
        let (mut sum_dist_same_clust, mut sum_dist_dif_clust): (f64, f64) = (0.0, 0.0);
        for (i, (row1, clust1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, clust2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = find_euclidean_distance(&row1, &row2);
                    if clust1 == clust2 {
                        sum_dist_same_clust += dist;
                        num_pairs_the_same_clust += 1.;
                    } else {
                        sum_dist_dif_clust += dist;
                        num_pairs_dif_clust += 1.;
                    }
                }
            }
        }
        Ok((sum_dist_same_clust / num_pairs_the_same_clust)
            / (sum_dist_dif_clust / num_pairs_dif_clust))
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

    #[pyfunction]
    fn gplus_index<'py>(
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

        let result = gplus_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn gplus_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let mut distances: Vec<f64> = Vec::default();
        let mut pairs_in_the_same_cluster: Vec<i8> = Vec::default();

        //calculating distances beetween pair of points and does they belong to the same cluster
        for (i, (row1, cluster1)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
            for (j, (row2, cluster2)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
                if i < j {
                    pairs_in_the_same_cluster.push((cluster1 != cluster2) as i8); // the same cluster =producer 0, different = 1
                    distances.push(find_euclidean_distance(&row1, &row2));
                }
            }
        }
        let mut s_minus = 0.0;

        // finding s_plus which represents the number of times a distance between two points
        // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
        // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
        //belonging to the same cluster

        for (i, (d1, b1)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
            for (j, (d2, b2)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
                if i < j && (*b1 == 0 && *b2 == 1) && d1 > d2 {
                    s_minus += 1.0;
                }
            }
        }

        let (n, _) = x.dim();
        let n_t = n as f64 * (n as f64 - 1.) / 2.0;
        Ok(2. * s_minus / (n_t * (n_t - 1.0)))
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

    #[pyfunction]
    fn ptbiserial_index<'py>(
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

        let result = ptbiserial_index_calc(x, y);
        result.map_err(PyValueError::new_err)
    }

    fn ptbiserial_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let (mut num_pairs_the_same_clust, mut num_pairs_dif_clust): (f64, f64) = (0., 0.);
        let (mut sum_dist_same_clust, mut sum_dist_dif_clust): (f64, f64) = (0.0, 0.0);
        let num_pairs_total = x.dim().0 as f64 * (x.dim().0 as f64 - 1.) / 2.;
        let distances: Array1<f64> = Array1::zeros(num_pairs_total as usize);
        for (i, (row1, clust1)) in zip(x.rows(), y).enumerate() {
            for (j, (row2, clust2)) in zip(x.rows(), y).enumerate() {
                if i < j {
                    let dist = find_euclidean_distance(&row1, &row2);

                    if clust1 == clust2 {
                        sum_dist_same_clust += dist;
                        num_pairs_the_same_clust += 1.;
                    } else {
                        sum_dist_dif_clust += dist;
                        num_pairs_dif_clust += 1.;
                    }
                }
            }
        }
        let std = distances.std(0.);

        let retval = (sum_dist_dif_clust / num_pairs_dif_clust
            - sum_dist_same_clust / num_pairs_the_same_clust)
            * (num_pairs_the_same_clust * num_pairs_dif_clust / num_pairs_total.powi(2)).sqrt()
            / std;
        Ok(retval)
    }
}
