use pyo3::prelude::*;

use ndarray::{
    parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    Array1, Array2, ArrayView1, ArrayView2, Axis, Zip,
};
use numpy::{npyffi::npy_int32, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{
    collections::{HashMap, HashSet},
    iter::zip,
};
pub mod indexes {
    use super::*;
    mod helper_funcs {
        use super::*;
        pub fn group(
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

        pub fn calc_clusters_centers(
            groups: &HashMap<i32, Array2<f64>>,
        ) -> HashMap<i32, Array1<f64>> {
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

        pub fn find_euclidean_distance(point1: &ArrayView1<f64>, point2: &ArrayView1<f64>) -> f64 {
            let sub_res = point2 - point1;
            f64::sqrt(sub_res.dot(&sub_res))
        }
        pub fn calc_covariance_matrix(x: Array2<f64>) -> Result<Array2<f64>, String> {
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
    }
    use helper_funcs::*;
    pub fn silhouette_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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
    pub fn davies_bouldin_index_calc(
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
    ) -> Result<f64, String> {
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
    pub fn calinski_harabasz_index_calc(
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
    ) -> Result<f64, String> {
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

    pub fn c_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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

    pub fn gamma_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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

    pub fn ball_hall_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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

    pub fn tau_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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
    pub fn dunn_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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
    pub fn sd_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<(f64, f64), String> {
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
    pub fn tracew_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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
    pub fn trcovw_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
        let cluster_centroids = calc_clusters_centers(&group(x, y)?);
        let (n, d) = x.dim();
        let mut diffs: Array2<f64> = Array2::zeros((n, d));
        for (i, (x, y)) in zip(x.rows(), y).enumerate() {
            let temp = cluster_centroids[y].to_owned() - x;
            diffs.row_mut(i).assign(&temp);
        }
        let w = diffs.t().dot(&diffs);
        let retval = helper_funcs::calc_covariance_matrix(w)?.diag().sum();

        Ok(retval)
    }
    pub fn ratkowsky_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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
    pub fn mcclain_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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
    pub fn gplus_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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

    pub fn ptbiserial_index_calc(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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

#[pymodule]
mod rust_ext {

    use super::*;
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
    macro_rules! wrapper {
        ($func:expr,$matrix:expr,$clusters:expr) => {{
            let x = $matrix.as_array();
            let y = $clusters.as_array();

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

            let result = $func(x, y);
            result.map_err(PyValueError::new_err)
        }};
    }

    #[pyfunction]
    pub fn silhouette_score<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::silhouette_index_calc, x, y)
    }
    #[pyfunction]
    pub fn davies_bouldin_score<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::davies_bouldin_index_calc, x, y)
    }
    #[pyfunction]
    pub fn calinski_harabasz_score<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::calinski_harabasz_index_calc, x, y)
    }
    #[pyfunction]
    pub fn c_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::c_index_calc, x, y)
    }
    #[pyfunction]
    pub fn gamma_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::gamma_index_calc, x, y)
    }

    #[pyfunction]
    pub fn ball_hall_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::ball_hall_index_calc, x, y)
    }
    #[pyfunction]
    pub fn tau_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::tau_index_calc, x, y)
    }
    #[pyfunction]
    pub fn dunn_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::dunn_index_calc, x, y)
    }
    #[pyfunction]
    pub fn sd_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<(f64, f64)> {
        wrapper!(indexes::sd_index_calc, x, y)
    }

    #[pyfunction]
    pub fn tracew_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::tracew_index_calc, x, y)
    }

    #[pyfunction]
    pub fn trcovw_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::trcovw_index_calc, x, y)
    }

    #[pyfunction]
    pub fn ratkowsky_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::ratkowsky_index_calc, x, y)
    }
    #[pyfunction]
    pub fn mcclain_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::mcclain_index_calc, x, y)
    }

    #[pyfunction]
    pub fn gplus_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::mcclain_index_calc, x, y)
    }
    #[pyfunction]
    pub fn ptbiserial_index<'py>(
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, npy_int32>,
    ) -> PyResult<f64> {
        wrapper!(indexes::ptbiserial_index_calc, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::*;
    use ndarray::{arr1, arr2};
    use std::time::Instant;
    fn initialize() -> (Array2<f64>, Array1<i32>) {
        (
            arr2(&[
                [-7.72642091, -8.39495682],
                [5.45339605, 0.74230537],
                [-2.97867201, 9.55684617],
                [6.04267315, 0.57131862],
                [-6.52183983, -6.31932507],
                [3.64934251, 1.40687195],
                [-2.17793419, 9.98983126],
                [4.42020695, 2.33028226],
                [4.73695639, 2.94181467],
                [-3.6601912, 9.38998415],
                [-3.05358035, 9.12520872],
                [-6.65216726, -5.57296684],
                [-6.35768563, -6.58312492],
                [-3.6155326, 7.8180795],
                [-1.77073104, 9.18565441],
                [-7.95051969, -6.39763718],
                [-6.60293639, -6.05292634],
                [-2.58120774, 10.01781903],
                [-7.76348463, -6.72638449],
                [-6.40638957, -6.95293851],
                [-2.97261532, 8.54855637],
                [-6.9567289, -6.53895762],
                [-7.32614214, -6.0237108],
                [-2.14780202, 10.55232269],
                [-2.54502366, 10.57892978],
                [-2.96983639, 10.07140835],
                [3.22450809, 1.55252436],
                [-6.25395984, -7.73726715],
                [-7.85430886, -6.09302499],
                [-8.1165779, -8.20056621],
                [-7.55965191, -6.6478559],
                [4.93599911, 2.23422496],
                [4.44751787, 2.27471703],
                [-5.72103161, -7.70079191],
                [-0.92998481, 9.78172086],
                [-3.10983631, 8.72259238],
                [-2.44166942, 7.58953794],
                [-2.18511365, 8.62920385],
                [5.55528095, 2.30192079],
                [4.73163961, -0.01439923],
                [-8.25729656, -7.81793463],
                [-2.98837186, 8.82862715],
                [4.60516707, 0.80449165],
                [-3.83738367, 9.21114736],
                [-2.62484591, 8.71318243],
                [3.57757512, 2.44676211],
                [-8.48711043, -6.69547573],
                [-6.70644627, -6.49479221],
                [-6.8666253, -5.42657552],
                [3.83138523, 1.47141264],
                [2.02013373, 2.79507219],
                [4.64499229, 1.73858255],
                [-1.6966718, 10.37052616],
                [-6.6197444, -6.09828672],
                [-6.05756703, -4.98331661],
                [-7.10308998, -6.1661091],
                [-3.52202874, 9.32853346],
                [-2.26723535, 7.10100588],
                [6.11777288, 1.45489947],
                [-4.23411546, 8.4519986],
                [-6.58655472, -7.59446101],
                [3.93782574, 1.64550754],
                [-7.12501531, -7.63384576],
                [2.72110762, 1.94665581],
                [-7.14428402, -4.15994043],
                [-6.66553345, -8.12584837],
                [4.70010905, 4.4364118],
                [-7.76914162, -7.69591988],
                [4.11011863, 2.48643712],
                [4.89742923, 1.89872377],
                [4.29716432, 1.17089241],
                [-6.62913434, -6.53366138],
                [-8.07093069, -6.22355598],
                [-2.16557933, 7.25124597],
                [4.7395302, 1.46969403],
                [-5.91625106, -6.46732867],
                [5.43091078, 1.06378223],
                [-6.82141847, -8.02307989],
                [6.52606474, 2.1477475],
                [3.08921541, 2.04173266],
                [-2.1475616, 8.36916637],
                [3.85662554, 1.65110817],
                [-1.68665271, 7.79344248],
                [-5.01385268, -6.40627667],
                [-2.52269485, 7.9565752],
                [-2.30033403, 7.054616],
                [-1.04354885, 8.78850983],
                [3.7204546, 3.52310409],
                [-3.98771961, 8.29444192],
                [4.24777068, 0.50965474],
                [4.7269259, 1.67416233],
                [5.78270165, 2.72510272],
                [-3.4172217, 7.60198243],
                [5.22673593, 4.16362531],
                [-3.11090424, 10.86656431],
                [-3.18611962, 9.62596242],
                [-1.4781981, 9.94556625],
                [4.47859312, 2.37722054],
                [-5.79657595, -5.82630754],
                [-3.34841515, 8.70507375],
            ]),
            arr1(&[
                1, 2, 0, 2, 1, 2, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 2, 1,
                1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 2, 2, 1, 0, 2, 0, 0, 2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 1,
                0, 0, 2, 0, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 0, 2, 1, 2, 1, 2, 2, 0, 2, 0, 1,
                0, 0, 0, 2, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 1, 0,
            ]),
        )
    }
    macro_rules! test_wrapper {
        ($func_name: expr, $val: expr) => {
            let (matrix, clusters) = initialize();

            let start = Instant::now();
            let res = $func_name(matrix.view(), clusters.view()).unwrap();
            let duration = start.elapsed();

            println!("Time elapsed is: {:?}", duration);

            assert_float_absolute_eq!(res, $val, 0.05);
        };
    }

    #[test]
    fn test_silhouette_score() {
        test_wrapper!(indexes::silhouette_index_calc, 0.8469881221532085);
    }

    #[test]
    fn test_davies_bouldin_score() {
        test_wrapper!(indexes::davies_bouldin_index_calc, 0.21374667882527568);
    }
    #[test]
    fn test_calinski_harabasz_score() {
        test_wrapper!(indexes::calinski_harabasz_index_calc, 1778.0880985088447);
    }
    #[test]
    fn test_c_index() {
        test_wrapper!(indexes::c_index_calc, 0.0);
    }
    #[test]
    fn test_gamma_index() {
        test_wrapper!(indexes::gamma_index_calc, 1.0);
    }
    #[test]
    fn test_ball_hall_index() {
        test_wrapper!(indexes::ball_hall_index_calc, 1.71928);
    }
    #[test]
    fn test_tau_index() {
        test_wrapper!(indexes::tau_index_calc, -1.316936e-05);
    }
    #[test]
    fn test_dunn_index() {
        test_wrapper!(indexes::dunn_index_calc, 1.320007);
    }
    #[test]
    fn test_sd_index() {
        let (matrix, clusters) = initialize();

        let start = Instant::now();
        let (sd_scat, sd_dis) = indexes::sd_index_calc(matrix.view(), clusters.view()).unwrap();
        let duration = start.elapsed();

        println!("Time elapsed is: {:?}", duration);

        assert_float_absolute_eq!(sd_scat, 0.02584332, 0.05);
        assert_float_absolute_eq!(sd_dis, 0.1810231, 0.05);
    }
    #[test]
    fn test_tracew_index() {
        test_wrapper!(indexes::tracew_index_calc, 171.911);
    }
    #[test]
    fn test_trcovw_index() {
        test_wrapper!(indexes::trcovw_index_calc, 70.47645);
    }
    #[test]
    fn test_ratkowsky_index() {
        test_wrapper!(indexes::ratkowsky_index_calc, 0.5692245);
    }
    #[test]
    fn test_mcclain_index() {
        test_wrapper!(indexes::mcclain_index_calc, 0.1243807);
    }
    #[test]
    fn test_gplus_index() {
        test_wrapper!(indexes::gplus_index_calc, 0.0);
    }
    #[test]
    fn test_ptbserial_index() {
        test_wrapper!(indexes::ptbiserial_index_calc, -5.571283);
    }
}
