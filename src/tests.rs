use crate::index_tree::IndexTreeBuilder;
use assert_float_eq::*;
use ndarray::{arr1, arr2, prelude::*};

const ACCURACY: f64 = 1e-5;
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
            1, 2, 0, 2, 1, 2, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 2, 1, 1,
            1, 1, 2, 2, 1, 0, 0, 0, 0, 2, 2, 1, 0, 2, 0, 0, 2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 1, 0, 0,
            2, 0, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 0, 2, 1, 2, 1, 2, 2, 0, 2, 0, 1, 0, 0, 0,
            2, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 1, 0,
        ]),
    )
}
#[test]
fn test_ball_hall_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_ball_hall().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.ball_hall.unwrap().unwrap().val, 1.71928, ACCURACY)
}
#[test]
fn test_silhouette_score() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());
    let tree = IndexTreeBuilder::default().add_silhouette().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(
        res.silhouette.unwrap().unwrap().val,
        0.8469881221532085,
        ACCURACY
    )
}
#[test]
fn test_davies_bouldin_score() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());
    let tree = IndexTreeBuilder::default().add_davies_bouldin().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(
        res.davies_bouldin.unwrap().unwrap().val,
        0.21374667882527568,
        ACCURACY
    )
}
#[test]
fn test_calinski_harabasz_score() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());
    let tree = IndexTreeBuilder::default().add_calinski_harabasz().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(
        res.calinski_harabasz.unwrap().unwrap().val,
        1778.0880985088447,
        ACCURACY
    )
}
#[test]
fn test_c_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());
    let tree = IndexTreeBuilder::default().add_c_index().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.c_index.unwrap().unwrap().val, 0.0, ACCURACY)
}
#[test]
fn test_gamma_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_gamma().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.gamma.unwrap().unwrap().val, 1.0, ACCURACY)
}

#[test]
fn test_rubin_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_rubin().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    // assert_float_absolute_eq!(res.rubin.unwrap().unwrap().val, 1099.786, ACCURACY)
    assert_float_absolute_eq!(
        res.rubin.unwrap().unwrap().val,
        1099.7859600203512,
        ACCURACY
    )
}
#[test]
fn test_mariott_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_mariott().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    // assert_float_absolute_eq!(res.mariott.unwrap().unwrap().val, 65223.64, ACCURACY)
    assert_float_absolute_eq!(
        res.mariott.unwrap().unwrap().val,
        65223.64251427408,
        ACCURACY
    )
}
#[test]
fn test_scott_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_scott().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    // assert_float_absolute_eq!(res.scott.unwrap().unwrap().val, 700.2871, ACCURACY)
    assert_float_absolute_eq!(res.scott.unwrap().unwrap().val, 700.2870858053287, ACCURACY)
}
#[test]
fn test_friedman_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_friedman().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.friedman.unwrap().unwrap().val, 70.47645, ACCURACY)
}
#[test]
fn test_tau_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());
    // let x = arr2(&[
    //     [2., 4.],
    //     [3., 5.],
    //     [8., 3.],
    //     [9., 2.],
    //     [10., 10.],
    //     [11., 12.],
    // ]);
    // let y = arr1(&[0, 0, 1, 1, 2, 2]);

    let tree = IndexTreeBuilder::default().add_tau().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x.view(), y.view()));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    // assert_float_absolute_eq!(res.tau.unwrap().unwrap().val, 0.6060916, ACCURACY)
    assert_float_absolute_eq!(res.tau.unwrap().unwrap().val, -1.316936e-05, ACCURACY)
}
#[test]
fn test_dunn_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_dunn().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.dunn.unwrap().unwrap().val, 1.320007, ACCURACY)
}
#[test]
fn test_tracew_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_tracew().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    // assert_float_absolute_eq!(res.tracew.unwrap().unwrap().val, 171.911, ACCURACY)
    assert_float_absolute_eq!(
        res.tracew.unwrap().unwrap().val,
        171.91098468600785,
        ACCURACY
    )
}
#[test]
fn test_gplus_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_gplus().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.gplus.unwrap().unwrap().val, 0.0, ACCURACY)
}
#[test]
fn test_ratkowsky_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_ratkowsky().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.ratkowsky.unwrap().unwrap().val, 0.5692245, ACCURACY)
}
#[test]
fn test_ptbserial_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_ptbiserial().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.ptbiserial.unwrap().unwrap().val, -5.571283, ACCURACY)
}
#[test]
fn test_mcclain_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_mcclain().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.mcclain.unwrap().unwrap().val, 0.1243807, ACCURACY)
}
//
//
//
//
//
//
//
#[test]
fn test_trcovw_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_trcovw().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(
        res.trcovw.unwrap().unwrap().val,
        3428.8903760801304,
        ACCURACY
    )
}
#[test]
fn test_sd_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_sd().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());

    assert_float_absolute_eq!(
        res.sd.unwrap().unwrap().val,
        0.02584332 + 0.1810231,
        ACCURACY
    )
}
#[test]
fn test_sdbw_index() {
    let (x, y) = initialize();
    let (x, y) = (x.view(), y.view());

    let tree = IndexTreeBuilder::default().add_sdbw().finish();

    let start = std::time::Instant::now();
    let res = tree.compute((x, y));
    let end = std::time::Instant::now();
    //
    println!("Duration {} milisecs", (end - start).as_millis());
    assert_float_absolute_eq!(res.sdbw.unwrap().unwrap().val, 0.02584332, ACCURACY)
}
