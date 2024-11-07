pub mod helper_funcs;

use std::{f64, ops::AddAssign};

use super::*;
use helper_funcs::*;
use itertools::Itertools;

pub fn silhouette_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let cluster_indexes = calc_clusters(&y);
    let mut temp: Vec<f64> = Vec::with_capacity(cluster_indexes.keys().len() - 1);
    let mut stor: Vec<f64> = Vec::with_capacity(x.nrows());
    for (c, arr) in cluster_indexes.iter() {
        for i in arr {
            let mut sum_inter_dists = 0.0;
            let row = x.row(*i);
            for j in arr {
                if i != j {
                    sum_inter_dists += find_euclidean_distance(&row.view(), &x.row(*j).view());
                }
            }
            for (c2, arr2) in cluster_indexes.iter() {
                if c2 != c {
                    let mut sum_intra_dists = 0.0;
                    for j2 in arr2 {
                        sum_intra_dists += find_euclidean_distance(&row.view(), &x.row(*j2).view());
                    }
                    temp.push(sum_intra_dists / arr2.len() as f64);
                }
            }
            let a = sum_inter_dists
                / if arr.len() == 1 {
                    1.0
                } else {
                    (arr.len() - 1) as f64
                };
            let b = temp
                .iter()
                .min_by(|a, b| a.total_cmp(b))
                .ok_or("Cant find min")?;

            stor.push((b - a) / a.max(*b));
            temp.clear()
        }
    }
    Ok(Array1::from_vec(stor).mean().ok_or("Cant calc mean")?)

    // let cluster_indexes = calc_clusters(&y);
    // let cluster_centers = calc_clusters_centers(&cluster_indexes, &x);
    //
    // let scores = Zip::from(x.rows())
    //     .and(y)
    //     .into_par_iter()
    //     .map(|(row, c)| {
    //         let nearest_cluster = cluster_centers
    //             .par_iter()
    //             .filter(|(i, _)| **i != *c)
    //             .map(|(i, row2)| (i, find_euclidean_distance(&row.view(), &row2.view())))
    //             .min_by(|(_, row2), (_, c2)| row2.total_cmp(c2))
    //             .map(|(i, _)| *i)
    //             .ok_or("Cant find closest cluster".to_string())?;
    //
    //         let nearest_cluster_distances = cluster_indexes
    //             .get(&(nearest_cluster))
    //             .ok_or("Cant get closest cluster")?
    //             .iter()
    //             .map(|iter| find_euclidean_distance(&row.view(), &x.row(*iter).view()))
    //             .collect::<Array1<f64>>();
    //
    //         let a: f64 = nearest_cluster_distances
    //             .mean()
    //             .ok_or("Cant calculate mean")?;
    //
    //         let point_cluster_distances = cluster_indexes
    //             .get(c)
    //             .ok_or("Cant get points cluster")?
    //             .iter()
    //             .map(|iter| find_euclidean_distance(&row.view(), &x.row(*iter).view()))
    //             .collect::<Array1<f64>>();
    //
    //         let b: f64 = point_cluster_distances
    //             .mean()
    //             .ok_or("Cant calculate mean")?;
    //
    //         Ok((a - b) / a.max(b))
    //     })
    //     .collect::<Result<Vec<f64>, String>>()?;
    //
    // Ok(Array1::from_vec(scores)
    //     .mean()
    //     .ok_or("Cant calculate mean")?)
}

pub fn davies_bouldin_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let clusters = calc_clusters(&y);
    let cluster_centers = calc_clusters_centers(&clusters, &x);

    let mut stor: HashMap<i32, f64> = HashMap::default();

    for (c, arr) in clusters.iter() {
        let temp = arr
            .par_iter()
            .map(|i| find_euclidean_distance(&cluster_centers[c].view(), &x.row(*i)))
            .sum::<f64>()
            / arr.len() as f64;
        stor.insert(*c, temp);
    }
    let q = clusters.keys().len();

    let mut acum = 0.0;
    let mut temp: Vec<f64> = Vec::with_capacity(q);
    for i in clusters.keys() {
        for j in clusters.keys() {
            if *i != *j {
                let coef = (stor[i] + stor[j])
                    / find_euclidean_distance(
                        &cluster_centers[i].view(),
                        &cluster_centers[j].view(),
                    );
                temp.push(coef);
            }
        }
        acum += temp
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .ok_or("Cant find max val")?;
        temp.clear();
    }

    Ok(acum / q as f64)
}

pub fn calinski_harabasz_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let number_of_objects = x.nrows() as f64;
    let data_center = x.mean_axis(Axis(0)).ok_or("Cant calc data centroid")?;
    let number_of_objects_in_clusters = y.iter().counts();
    let cluster_indexes = calc_clusters(&y);
    let cluster_centers = calc_clusters_centers(&cluster_indexes, &x);
    let number_of_clusters = cluster_indexes.keys().len() as f64;

    let inbetween_group_dispersion = cluster_centers
        .iter()
        .map(|(i, c)| {
            find_euclidean_distance(&c.view(), &data_center.view()).powi(2)
                * number_of_objects_in_clusters[i] as f64
        })
        .sum::<f64>();

    let within_group_dispersion = cluster_indexes
        .iter()
        .map(|(c, arr)| {
            arr.iter()
                .map(|i| find_euclidean_distance(&cluster_centers[c].view(), &x.row(*i)).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>();
    let res = (inbetween_group_dispersion / (number_of_clusters - 1.0))
        / (within_group_dispersion / (number_of_objects - number_of_clusters));

    Ok(res)
}

pub fn c_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    //calculating Nw  -- total number of pairs of observations belonging to the same cluster
    let counts = y
        .iter()
        .counts()
        .iter()
        .map(|(i, n)| (**i, (*n * (*n - 1)) / 2))
        .collect::<HashMap<i32, usize>>();
    let number_of_pairs_in_clusters = counts.values().sum::<usize>();

    //calculating distances beetween all possible pars of points in dataset
    let mut distances = Vec::with_capacity(y.len() * (y.len() - 1) / 2);
    let mut distances_per_cluster = counts
        .keys()
        .map(|i| (*i, 0.0))
        .collect::<HashMap<i32, f64>>();

    for (i, (row1, c1)) in zip(x.rows(), y).enumerate() {
        for (j, (row2, c2)) in zip(x.rows(), y).enumerate() {
            if i < j {
                let dist = find_euclidean_distance(&row1, &row2);
                distances.push(dist);
                if c1 == c2 {
                    distances_per_cluster
                        .get_mut(c1)
                        .ok_or("cant add distance")?
                        .add_assign(dist);
                }
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
    let sum_of_withincluster_distances = distances_per_cluster.values().sum::<f64>();

    //calculating c_index value
    Ok((sum_of_withincluster_distances - sum_of_minimum_distances)
        / (sum_of_maximum_distances - sum_of_minimum_distances))
}
pub fn gamma_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let n = y.len() * (y.len() - 1) / 2;
    let mut distances: Vec<f64> = Vec::with_capacity(n);
    let mut pairs_in_the_same_cluster: Vec<i8> = Vec::with_capacity(n);

    //calculating distances beetween pair of points and does they belong to the same cluster
    for (i, (row1, cluster1)) in zip(x.rows(), y).enumerate() {
        for (j, (row2, cluster2)) in zip(x.rows(), y).enumerate() {
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

pub fn ball_hall_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let clusters = calc_clusters(&y);
    let number_of_cluster = clusters.keys().len();
    let clusters_centroids = calc_clusters_centers(&clusters, &x);

    let std = clusters
        .par_iter()
        .map(|(c, arr)| {
            arr.iter()
                .map(|i| {
                    find_euclidean_distance(&clusters_centroids[c].view(), &x.row(*i).view())
                        .powi(2)
                })
                .sum::<f64>()
                / arr.len() as f64
        })
        .sum::<f64>();

    Ok(std / number_of_cluster as f64)
}
//
// pub fn tau_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
//     let mut distances: Vec<f64> = Vec::default();
//     let mut pairs_in_the_same_cluster: Vec<i8> = Vec::default();
//
//     //calculating distances beetween pair of points and does they belong to the same cluster
//     for (i, (row1, cluster1)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
//         for (j, (row2, cluster2)) in x.axis_iter(Axis(0)).zip(y).enumerate() {
//             if i < j {
//                 pairs_in_the_same_cluster.push((cluster1 != cluster2) as i8); // the same cluster = 0, different = 1
//                 distances.push(find_euclidean_distance(&row1, &row2));
//             }
//         }
//     }
//
//     let number_of_pairs_in_the_same_cluster = pairs_in_the_same_cluster
//         .iter()
//         .filter(|i| **i == 1)
//         .count();
//
//     let total_number_of_pairs = x.len() * (x.len() - 1) / 2;
//
//     let number_of_pairs_in_different_clusters =
//         total_number_of_pairs - number_of_pairs_in_the_same_cluster;
//
//     let (mut s_plus, mut s_minus) = (0.0, 0.0);
//
//     // finding s_plus which represents the number of times a distance between two points
//     // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
//     // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
//     //belonging to the same cluster
//
//     let mut ties: HashMap<u64, i32> = HashMap::default();
//
//     for (i, (d1, b1)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
//         for (j, (d2, b2)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
//             if i < j && (*b1 == 0 && *b2 == 1) {
//                 if d1 < d2 {
//                     s_plus += 1.0;
//                 }
//                 if d1 > d2 {
//                     s_minus += 1.0;
//                 }
//                 if d1 == d2 {
//                     ties.entry(d1.to_bits())
//                         .and_modify(|val| *val += 1)
//                         .or_insert(0);
//                 }
//             }
//         }
//     }
//
//     let v0 = (total_number_of_pairs * (total_number_of_pairs - 1)) as f64 / 2.0;
//     let v1 = ties
//         .values()
//         .map(|c| (c * (c - 1)) as f64 / 2.0)
//         .sum::<f64>();
//     let v2 = (number_of_pairs_in_different_clusters * (number_of_pairs_in_different_clusters - 1))
//         as f64
//         / 2.0
//         + (number_of_pairs_in_the_same_cluster * (number_of_pairs_in_the_same_cluster - 1)) as f64
//             / 2.0;
//
//     Ok((s_plus - s_minus) / f64::sqrt((v0 - v1) * (v0 - v2)))
// }

pub fn tau_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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

    let total_number_of_pairs = x.len() * (x.len() - 1) / 2;
    let (mut s_plus, mut s_minus): (usize, usize) = (0, 0);

    // finding s_plus which represents the number of times a distance between two points
    // which belong to the same cluster is strictly smaller than the distance between two points not belonging to the same cluster
    // and s_minus which represents the number of times distance between two points lying in the same cluster  is strictly greater than a distance between two points not
    // belonging to the same cluster
    let mut t: usize = 0;
    for (i, (d1, b1)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
        for (j, (d2, b2)) in zip(&distances, &pairs_in_the_same_cluster).enumerate() {
            if i < j {
                if (*b1 == 1 && *b2 == 1) || (*b1 == 0 && *b2 == 0) {
                    t += 1;
                }
                if *b1 == 0 && *b2 == 1 {
                    s_plus += (d1 < d2) as u8 as usize;
                    s_minus += (d1 > d2) as u8 as usize;
                }
            }
        }
    }
    let v0 = (total_number_of_pairs * (total_number_of_pairs - 1)) as f64 / 2.0;
    Ok((s_plus - s_minus) as f64 / ((v0 - t as f64) * v0).sqrt())
}

pub fn dunn_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let n = y.len() * (y.len() - 1) / 2;
    let mut intercluster_distances: Vec<f64> = Vec::with_capacity(n);
    let mut intracluster_distances: Vec<f64> = Vec::with_capacity(n);

    for (i, (row1, cluster1)) in zip(x.rows(), y).enumerate() {
        for (j, (row2, cluster2)) in zip(x.rows(), y).enumerate() {
            if i < j {
                let dist = find_euclidean_distance(&row1, &row2);
                if cluster1 == cluster2 {
                    intracluster_distances.push(dist);
                } else {
                    intercluster_distances.push(dist);
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
pub fn sd_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<(f64, f64), String> {
    let features_variances = x.var_axis(Axis(0), 0.);
    let clusters = calc_clusters(&y);
    let cluster_centers = calc_clusters_centers(&clusters, &x);

    let mut stor: Vec<f64> = Vec::with_capacity(clusters.keys().len());

    for (c, arr) in clusters {
        // let mut cluster: Array2<f64> = Array2::zeros((arr.len(), x.ncols()));
        // for (i, j) in arr.iter().enumerate() {
        //     cluster.row_mut(i).add_assign(&x.row(*j));
        // }
        // stor.push(calc_vector_euclidean_length(
        //     &cluster.var_axis(Axis(0), 0.).view(),
        // ));

        let mut row = Array1::zeros(x.ncols());
        for i in arr.iter() {
            row.add_assign(&(&x.row(*i) - &cluster_centers[&c]).powi(2));
        }
        row /= arr.len() as f64;
        stor.push(calc_vector_euclidean_length(&row.view()));
    }
    let S = Array1::from_vec(stor).mean().ok_or("Cant calculate mean")?
        / calc_vector_euclidean_length(&features_variances.view());

    let D = |cluster_centroids: &HashMap<i32, Array1<f64>>| -> f64 {
        let mut d = 0.0;
        let mut d_max = f64::MIN;
        let mut d_min = f64::MAX;
        for (i, row1) in cluster_centroids {
            let mut dist_acum = 0.0;
            for (j, row2) in cluster_centroids {
                if i != j {
                    let dist = find_euclidean_distance(&row1.view(), &row2.view());
                    dist_acum += dist;
                    if i < j {
                        if dist > d_max {
                            d_max = dist;
                        }
                        if dist < d_min {
                            d_min = dist;
                        }
                    }
                }
            }
            if dist_acum != 0.0 {
                d += 1. / dist_acum;
            }
        }
        d * d_max / d_min
    };

    Ok((S, D(&cluster_centers)))
}

pub fn sdbw_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let features_variances = x.var_axis(Axis(0), 0.);
    let clusters = calc_clusters(&y);
    let cluster_centers = calc_clusters_centers(&clusters, &x);

    let mut stor: Vec<f64> = Vec::with_capacity(clusters.len());
    let mut stor_sqrt: Vec<f64> = Vec::with_capacity(clusters.len());
    for (c, arr) in clusters.iter() {
        // let mut cluster: Array2<f64> = Array2::zeros((arr.len(), x.ncols()));
        // for (i, j) in arr.iter().enumerate() {
        //     cluster.row_mut(i).add_assign(&x.row(*j));
        // }
        // stor.push(calc_vector_euclidean_length(
        //     &cluster.var_axis(Axis(0), 0.).view(),
        // ));

        let mut row = Array1::zeros(x.ncols());
        for i in arr.iter() {
            row.add_assign(&(&x.row(*i) - &cluster_centers[c]).powi(2));
        }
        row /= arr.len() as f64;
        let vec_length = calc_vector_euclidean_length(&row.view());
        stor.push(vec_length);
        stor_sqrt.push(vec_length.sqrt());
    }
    let S = Array1::from_vec(stor).mean().ok_or("Cant calculate mean")?
        / calc_vector_euclidean_length(&features_variances.view());

    let std = Array1::from_vec(stor_sqrt).mean().ok_or("Cant calc std")?;
    let dencity = |cluster1: i32, cluster2: i32, point: &ArrayView1<f64>| -> usize {
        let mut retval: usize = 0;
        for i in [cluster1, cluster2] {
            for j in clusters[&i].iter() {
                let dist = find_euclidean_distance(point, &x.row(*j).view());
                retval += (dist <= std) as u8 as usize;
            }
        }
        retval
    };
    let mut acum: usize = 0;
    for (i, c1) in clusters.keys().enumerate() {
        for (j, c2) in clusters.keys().enumerate() {
            if i < j {
                let R = dencity(
                    *c1,
                    *c2,
                    &((&cluster_centers[c1] + &cluster_centers[c2]) / 2.0).view(),
                ) / dencity(*c1, *c2, &cluster_centers[c1].view()).max(dencity(
                    *c1,
                    *c2,
                    &cluster_centers[c2].view(),
                ));
                acum += R;
            }
        }
    }

    let q = clusters.keys().len();
    let Dbw = ((1 / (q * (q - 1))) * acum) as f64;
    Ok(S + Dbw)
}

pub fn tracew_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let cluster_centroids = calc_clusters_centers(&calc_clusters(&y), &x);
    let (n, d) = x.dim();
    let mut diffs: Array2<f64> = Array2::zeros((n, d));
    for (i, (x, y)) in zip(x.rows(), y).enumerate() {
        let temp = &cluster_centroids[y] - &x;
        diffs.row_mut(i).assign(&temp);
    }

    let retval = diffs.t().dot(&diffs).diag().sum();
    Ok(retval)
}
pub fn trcovw_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let cluster_centroids = calc_clusters_centers(&calc_clusters(&y), &x);
    let mut diffs: Array2<f64> = Array2::zeros((x.nrows(), x.ncols()));
    for (i, (x, y)) in zip(x.rows(), y).enumerate() {
        let temp = &cluster_centroids[y] - &x;
        diffs.row_mut(i).assign(&temp);
    }
    let w = diffs.t().dot(&diffs);

    let w_norm = &w - &w.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
    let n = w.nrows() as f64;
    let retval = (w_norm.t().dot(&w_norm) / n).diag().sum();
    Ok(retval)
}
pub fn ratkowsky_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let x_mean = x
        .mean_axis(Axis(0))
        .ok_or("Cant compute mean for dataset")?;
    let clusters = calc_clusters(&y);
    let clusters_centroids = calc_clusters_centers(&clusters, &x);

    let (num_of_elems, num_of_vars) = x.dim();

    let mut bgss: Array1<f64> = Array1::zeros(num_of_vars);
    for (i, c) in clusters_centroids {
        bgss = bgss + clusters[&i].len() as f64 * (c - &x_mean).pow2();
    }

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
pub fn gplus_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
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

pub fn ptbiserial_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let (mut num_pairs_the_same_clust, mut num_pairs_dif_clust): (usize, usize) = (0, 0);
    let (mut sum_dist_same_clust, mut sum_dist_dif_clust): (f64, f64) = (0.0, 0.0);
    let num_pairs_total = x.nrows() * (x.nrows() - 1) / 2;
    let mut distances: Array1<f64> = Array1::zeros(num_pairs_total);
    let mut ctr = 0;
    for (i, (row1, clust1)) in zip(x.rows(), y).enumerate() {
        for (j, (row2, clust2)) in zip(x.rows(), y).enumerate() {
            if i < j {
                let dist = find_euclidean_distance(&row1, &row2);
                distances[ctr] = dist;
                if clust1 == clust2 {
                    sum_dist_same_clust += dist;
                    num_pairs_the_same_clust += 1;
                } else {
                    sum_dist_dif_clust += dist;
                    num_pairs_dif_clust += 1;
                }
                ctr += 1;
            }
        }
    }
    let std = distances.std(0.);

    let (num_pairs_the_same_clust, num_pairs_dif_clust, num_pairs_total) = (
        num_pairs_the_same_clust as f64,
        num_pairs_dif_clust as f64,
        num_pairs_total as f64,
    );

    // let retval = (sum_dist_dif_clust / num_pairs_dif_clust
    //     - sum_dist_same_clust / num_pairs_the_same_clust)
    //     * (num_pairs_the_same_clust * num_pairs_dif_clust / num_pairs_total.powi(2)).sqrt()
    //     / std;

    let retval = (sum_dist_same_clust / num_pairs_the_same_clust
        - sum_dist_dif_clust / num_pairs_dif_clust)
        * (num_pairs_dif_clust * num_pairs_the_same_clust).sqrt()
        / num_pairs_total;

    Ok(retval)
}

pub fn scott_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let cluster_centroids = calc_clusters_centers(&calc_clusters(&y), &x);
    let x_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
    let mut diffs1: Array2<f64> = Array2::zeros(x.dim());
    let mut diffs2: Array2<f64> = Array2::zeros(x.dim());
    for (i, (x, y)) in zip(x.rows(), y).enumerate() {
        diffs1.row_mut(i).assign(&(&x - &cluster_centroids[y]));
        diffs2.row_mut(i).assign(&(&x - &x_mean));
    }

    let w_q = diffs1.t().dot(&diffs1);
    let t = diffs2.t().dot(&diffs2);
    let det_t = calc_matrix_determinant(&t.view())?;
    let det_w_q = calc_matrix_determinant(&w_q.view())?;
    Ok(x.nrows() as f64 * (det_t / det_w_q).ln())
}

pub fn mariott_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let cluster_centroids = calc_clusters_centers(&calc_clusters(&y), &x);
    let mut diffs: Array2<f64> = Array2::zeros(x.dim());
    for (i, (x, y)) in zip(x.rows(), y).enumerate() {
        diffs.row_mut(i).assign(&(&x - &cluster_centroids[y]));
    }

    let w_q = diffs.t().dot(&diffs);
    let det_w_q = calc_matrix_determinant(&w_q.view())?;
    let q = cluster_centroids.keys().len() as f64;
    Ok(q.powi(2) * det_w_q)
}

pub fn rubin_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    let cluster_centroids = calc_clusters_centers(&calc_clusters(&y), &x);
    let x_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
    let mut diffs1: Array2<f64> = Array2::zeros(x.dim());
    let mut diffs2: Array2<f64> = Array2::zeros(x.dim());
    for (i, (x, y)) in zip(x.rows(), y).enumerate() {
        diffs1.row_mut(i).assign(&(&x - &cluster_centroids[y]));
        diffs2.row_mut(i).assign(&(&x - &x_mean));
    }

    let w_q = diffs1.t().dot(&diffs1);
    let t = diffs2.t().dot(&diffs2);
    let det_t = calc_matrix_determinant(&t.view())?;
    let det_w_q = calc_matrix_determinant(&w_q.view())?;
    Ok(det_t / det_w_q)
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
//
//
//
//
//
//
//

pub fn duda_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    if y.iter().counts().keys().len() != 2 {
        return Err("There is more than 2 clusters".to_string());
    }

    let dataset_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
    let clusters = calc_clusters(&y);
    let cluster_centers = calc_clusters_centers(&clusters, &x);

    let within_group_dispersion_parent = {
        let diff = &x - &dataset_mean;
        diff.dot(&diff.t()).diag().sum()
    };

    let mut within_group_dispersion_children: HashMap<i32, f64> = HashMap::default();
    for (cl_idx, idxs) in clusters.iter() {
        for idx in idxs {
            let diff = &x.row(*idx) - &cluster_centers[cl_idx];

            within_group_dispersion_children
                .entry(*cl_idx)
                .and_modify(|v| *v += diff.dot(&diff))
                .or_insert(0.);
        }
    }

    Ok(within_group_dispersion_parent / (within_group_dispersion_children.values().sum::<f64>()))
}

pub fn pseudot2_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    if y.iter().counts().keys().len() != 2 {
        return Err("There is more than 2 clusters".to_string());
    }
    let dataset_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
    let clusters = calc_clusters(&y);
    let cluster_centers = calc_clusters_centers(&clusters, &x);

    let within_group_dispersion_parent = {
        let diff = &x - &dataset_mean;
        diff.dot(&diff.t()).diag().sum()
    };

    let mut within_group_dispersion_children: HashMap<i32, f64> = HashMap::default();
    for (cl_idx, idxs) in clusters.iter() {
        for idx in idxs {
            let diff = &x.row(*idx) - &cluster_centers[cl_idx];

            within_group_dispersion_children
                .entry(*cl_idx)
                .and_modify(|v| *v += diff.dot(&diff))
                .or_insert(0.);
        }
    }
    let w_children_sum = within_group_dispersion_children.values().sum::<f64>();
    let retval =
        (within_group_dispersion_parent - w_children_sum) / w_children_sum / (x.nrows() - 2) as f64;
    Ok(retval)
}

pub fn beale_index(x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<f64, String> {
    if y.iter().counts().keys().len() != 2 {
        return Err("There is more than 2 clusters".to_string());
    }
    let dataset_mean = x.mean_axis(Axis(0)).ok_or("Cant calc mean")?;
    let clusters = calc_clusters(&y);
    let cluster_centers = calc_clusters_centers(&clusters, &x);

    let within_group_dispersion_parent = {
        let diff = &x - &dataset_mean;
        diff.dot(&diff.t()).diag().sum()
    };

    let mut within_group_dispersion_children: HashMap<i32, f64> = HashMap::default();
    for (cl_idx, idxs) in clusters.iter() {
        for idx in idxs {
            let diff = &x.row(*idx) - &cluster_centers[cl_idx];

            within_group_dispersion_children
                .entry(*cl_idx)
                .and_modify(|v| *v += diff.dot(&diff))
                .or_insert(0.);
        }
    }
    let n = x.nrows();
    let w_children_sum = within_group_dispersion_children.values().sum::<f64>();
    let retval = ((within_group_dispersion_parent - w_children_sum) / w_children_sum)
        / (2.0 * (((n - 1) as f64) / ((n - 2) as f64)).powf(2.0 / x.ncols() as f64) - 1.0);
    Ok(retval)
}
