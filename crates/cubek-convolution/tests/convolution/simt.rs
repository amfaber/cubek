//! The SIMT convolution (forward, input gradient, weight gradient) against the definition of a
//! convolution computed on the host.
//!
//! The data are small integers on short periods, so every partial sum is exact in f32 and any
//! summation order (the split tilings sum their slices last) gives the same result: the
//! comparisons are exact. Each case runs with every tiling (the narrow one must accept it; the
//! others may decline) and with contiguous and NCHW-strided inputs.

use cubecl::{
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubek_convolution::{
    ConvolutionArgs, SimtBackwardDataTensors, SimtBackwardWeightTensors, SimtTensors, SimtTiling,
    components::ConvSetupError, launch_simt, launch_simt_backward_data,
    launch_simt_backward_weight,
};
use cubek_test_utils::{HostData, HostDataType, TestInput};

const TILINGS: [SimtTiling; 3] = [
    SimtTiling::NARROW,
    SimtTiling::NARROW_SPLIT,
    SimtTiling::WIDE,
];

#[derive(Debug, Clone, Copy)]
struct Case {
    name: &'static str,
    b: usize,
    c_in: usize,
    c_out: usize,
    in_hw: [usize; 2],
    k: [usize; 2],
    stride: usize,
    /// At both ends of each spatial dimension.
    padding: usize,
    dilation: usize,
}

#[allow(clippy::too_many_arguments)]
const fn case(
    name: &'static str,
    b: usize,
    c_in: usize,
    c_out: usize,
    in_hw: [usize; 2],
    k: [usize; 2],
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Case {
    Case {
        name,
        b,
        c_in,
        c_out,
        in_hw,
        k,
        stride,
        padding,
        dilation,
    }
}

/// Output channels filling one tile, two, and a tile cut short; positions filling no tile and
/// several; stride, dilation, padding; a reduction over several taps per stage, and a long one
/// over few positions, which the split tiling splits.
const CASES: [Case; 8] = [
    case("plain", 2, 16, 64, [7, 6], [3, 3], 1, 0, 1),
    case("padded, two tiles", 3, 32, 128, [9, 9], [3, 3], 1, 1, 1),
    case("channels cut short", 1, 16, 36, [8, 7], [3, 2], 1, 1, 1),
    case("dilated", 2, 16, 32, [10, 9], [3, 3], 1, 2, 2),
    case("many positions", 4, 16, 32, [40, 40], [3, 3], 1, 1, 1),
    case("deep, few positions", 2, 64, 64, [5, 5], [3, 3], 1, 1, 1),
    case("twelve channels", 2, 12, 32, [9, 8], [3, 3], 1, 1, 1),
    case("strided", 3, 16, 64, [11, 10], [3, 3], 2, 1, 1),
];

/// Forward only: a pointwise strided layer, and input channels short of a vector (the RGB stem
/// of a ResNet), which the forward pass pads with zero channels.
const FORWARD_ONLY: [Case; 3] = [
    case("pointwise strided", 2, 32, 64, [8, 8], [1, 1], 2, 0, 1),
    case("rgb stem", 2, 3, 64, [20, 19], [7, 7], 2, 3, 1),
    case("four-channel stem", 2, 4, 64, [20, 19], [7, 7], 2, 3, 1),
];

impl Case {
    fn out_hw(&self) -> [usize; 2] {
        [0, 1].map(|i| {
            (self.in_hw[i] + 2 * self.padding - self.dilation * (self.k[i] - 1) - 1) / self.stride
                + 1
        })
    }

    fn x_shape(&self) -> [usize; 4] {
        [self.b, self.in_hw[0], self.in_hw[1], self.c_in]
    }

    fn w_shape(&self) -> [usize; 4] {
        [self.c_out, self.k[0], self.k[1], self.c_in]
    }

    fn y_shape(&self) -> [usize; 4] {
        let [oh, ow] = self.out_hw();
        [self.b, oh, ow, self.c_out]
    }

    fn args(&self) -> ConvolutionArgs<2> {
        ConvolutionArgs {
            stride: [self.stride; 2],
            padding: [self.padding; 2],
            dilation: [self.dilation; 2],
        }
    }

    /// `f(x index, w index, y index)` for every product of the convolution, all NHWC (the weight
    /// `[c_out, kh, kw, c_in]`).
    fn for_each_product(&self, mut f: impl FnMut(usize, usize, usize)) {
        let [ih, iw] = self.in_hw;
        let [oh, ow] = self.out_hw();
        for b in 0..self.b {
            for y in 0..oh {
                for x in 0..ow {
                    for ky in 0..self.k[0] {
                        let iy =
                            (y * self.stride + ky * self.dilation) as isize - self.padding as isize;
                        for kx in 0..self.k[1] {
                            let ix = (x * self.stride + kx * self.dilation) as isize
                                - self.padding as isize;
                            if iy < 0 || ix < 0 || iy >= ih as isize || ix >= iw as isize {
                                continue;
                            }
                            let x_at = ((b * ih + iy as usize) * iw + ix as usize) * self.c_in;
                            let w_at = (ky * self.k[1] + kx) * self.c_in;
                            let y_at = ((b * oh + y) * ow + x) * self.c_out;
                            for co in 0..self.c_out {
                                for ci in 0..self.c_in {
                                    f(
                                        x_at + ci,
                                        co * self.k[0] * self.k[1] * self.c_in + w_at + ci,
                                        y_at + co,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Small integers on a short period: products at most 4, partial sums far below 2^24.
fn ramp(n: usize, period: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % period) as f32) - 1.0).collect()
}

fn client() -> cubecl::client::Client {
    cubecl::test_device().client()
}

fn f32_type() -> ElemType {
    f32::elem_type_native()
}

fn upload(data: &[f32], shape: [usize; 4]) -> TensorHandle {
    TestInput::builder(client(), Shape::new(shape))
        .dtype(f32_type())
        .custom(data.to_vec())
        .generate_without_host_data()
}

/// `data` (NHWC) as a binding of shape `[n, h, w, c]`: contiguous, or with NCHW strides over a
/// tensor laid out channels-first, as a framework that keeps NCHW tensors passes them.
fn nhwc(data: &[f32], [n, h, w, c]: [usize; 4], contiguous: bool) -> (TensorHandle, TensorBinding) {
    if contiguous {
        let handle = upload(data, [n, h, w, c]);
        let binding = handle.clone().binding();
        return (handle, binding);
    }
    let mut nchw = vec![0.0; data.len()];
    for b in 0..n {
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    nchw[((b * c + ch) * h + y) * w + x] = data[((b * h + y) * w + x) * c + ch];
                }
            }
        }
    }
    let handle = upload(&nchw, [n, c, h, w]);
    let mut binding = handle.clone().binding();
    binding.shape = Shape::new([n, h, w, c]);
    binding.strides = Strides::new(&[c * h * w, w, 1, h * w]);
    (handle, binding)
}

/// A contiguous tensor filled with a value no result reaches, so a cell left unwritten fails.
fn output(shape: [usize; 4]) -> TensorHandle {
    upload(&vec![4096.0; shape.iter().product()], shape)
}

fn read(handle: TensorHandle) -> Vec<f32> {
    let host = HostData::from_tensor_handle(&client(), handle, HostDataType::F32);
    let len = host.shape.iter().product::<usize>();
    (0..len).map(|i| host.data.get_f32(i)).collect()
}

fn compare(what: &str, got: &[f32], want: &[f32]) -> Result<(), String> {
    match got.iter().zip(want).position(|(g, w)| g != w) {
        None => Ok(()),
        Some(i) => Err(format!("{what}: at {i}, got {} want {}", got[i], want[i])),
    }
}

/// Runs `check` for each tiling, and fails on any mismatch, or on a decline of the narrow tiling.
fn for_each_tiling(
    cases: &[Case],
    check: impl Fn(&Case, SimtTiling) -> Result<Result<(), String>, ConvSetupError>,
) {
    let mut bad = Vec::new();
    let mut ran = [0; TILINGS.len()];
    for case in cases {
        for (t, tiling) in TILINGS.into_iter().enumerate() {
            match check(case, tiling) {
                Ok(Ok(())) => ran[t] += 1,
                Ok(Err(mismatch)) => bad.push(format!("{} {tiling:?}: {mismatch}", case.name)),
                Err(err) if tiling == SimtTiling::NARROW => {
                    bad.push(format!("{} {tiling:?}: declined: {err:?}", case.name))
                }
                Err(_) => {}
            }
        }
    }
    assert!(bad.is_empty(), "failing cases:\n{}", bad.join("\n"));
    // A tiling that declined everything would have tested nothing.
    for (tiling, ran) in TILINGS.iter().zip(ran) {
        assert!(ran > 0, "{tiling:?} ran no case");
    }
}

fn check_forward(case: &Case, tiling: SimtTiling) -> Result<Result<(), String>, ConvSetupError> {
    let x = ramp(case.x_shape().iter().product(), 4);
    let w = ramp(case.w_shape().iter().product(), 3);
    let bias = ramp(case.c_out, 5);
    for (with_bias, contiguous) in [(false, true), (true, true), (true, false)] {
        let mut want = vec![0.0; case.y_shape().iter().product()];
        if with_bias {
            for (i, v) in want.iter_mut().enumerate() {
                *v = bias[i % case.c_out];
            }
        }
        case.for_each_product(|xi, wi, yi| want[yi] += x[xi] * w[wi]);

        let (_x, x_binding) = nhwc(&x, case.x_shape(), contiguous);
        let (_w, w_binding) = nhwc(&w, case.w_shape(), contiguous);
        let bias_handle = TestInput::builder(client(), Shape::new([case.c_out]))
            .dtype(f32_type())
            .custom(bias.clone())
            .generate_without_host_data();
        let out = output(case.y_shape());
        launch_simt(
            &client(),
            SimtTensors {
                input: x_binding,
                weight: w_binding,
                bias: with_bias.then(|| bias_handle.binding()),
                out: out.clone().binding(),
            },
            case.args(),
            1,
            f32_type(),
            tiling,
        )?;
        let label = format!("bias {with_bias}, contiguous {contiguous}");
        if let Err(mismatch) = compare(&label, &read(out), &want) {
            return Ok(Err(mismatch));
        }
    }
    Ok(Ok(()))
}

fn check_input_gradient(
    case: &Case,
    tiling: SimtTiling,
) -> Result<Result<(), String>, ConvSetupError> {
    let w = ramp(case.w_shape().iter().product(), 3);
    let dy = ramp(case.y_shape().iter().product(), 4);
    let mut want = vec![0.0; case.x_shape().iter().product()];
    case.for_each_product(|xi, wi, yi| want[xi] += dy[yi] * w[wi]);
    for contiguous in [true, false] {
        let (_dy, dy_binding) = nhwc(&dy, case.y_shape(), contiguous);
        let (_w, w_binding) = nhwc(&w, case.w_shape(), contiguous);
        let dx = output(case.x_shape());
        launch_simt_backward_data(
            &client(),
            SimtBackwardDataTensors {
                out_grad: dy_binding,
                weight: w_binding,
                in_grad: dx.clone().binding(),
            },
            case.args(),
            1,
            f32_type(),
            tiling,
        )?;
        if let Err(mismatch) = compare(&format!("contiguous {contiguous}"), &read(dx), &want) {
            return Ok(Err(mismatch));
        }
    }
    Ok(Ok(()))
}

fn check_weight_gradient(case: &Case) -> Result<Result<(), String>, ConvSetupError> {
    let x = ramp(case.x_shape().iter().product(), 4);
    let dy = ramp(case.y_shape().iter().product(), 3);
    let mut want = vec![0.0; case.w_shape().iter().product()];
    case.for_each_product(|xi, wi, yi| want[wi] += x[xi] * dy[yi]);
    let [c_out, kh, kw, c_in] = case.w_shape();
    for contiguous in [true, false] {
        let (_x, x_binding) = nhwc(&x, case.x_shape(), contiguous);
        let (_dy, dy_binding) = nhwc(&dy, case.y_shape(), contiguous);
        // Laid out `[kh, kw, c_in, c_out]`, seen as `[c_out, kh, kw, c_in]`.
        let dw = output([kh, kw, c_in, c_out]);
        let mut dw_binding = dw.clone().binding();
        dw_binding.shape = Shape::new([c_out, kh, kw, c_in]);
        dw_binding.strides = Strides::new(&[1, kw * c_in * c_out, c_in * c_out, c_out]);
        launch_simt_backward_weight(
            &client(),
            SimtBackwardWeightTensors {
                input: x_binding,
                out_grad: dy_binding,
                weight_grad: dw_binding,
            },
            case.args(),
            1,
            f32_type(),
        )?;
        let got = read(dw);
        // Back to `[c_out, kh, kw, c_in]`.
        let mut got_ohwi = vec![0.0; got.len()];
        for co in 0..c_out {
            for tap in 0..kh * kw {
                for ci in 0..c_in {
                    got_ohwi[(co * kh * kw + tap) * c_in + ci] =
                        got[(tap * c_in + ci) * c_out + co];
                }
            }
        }
        if let Err(mismatch) = compare(&format!("contiguous {contiguous}"), &got_ohwi, &want) {
            return Ok(Err(mismatch));
        }
    }
    Ok(Ok(()))
}

#[test]
fn simt_forward_matches_the_definition() {
    let mut cases = CASES.to_vec();
    cases.extend(FORWARD_ONLY);
    for_each_tiling(&cases, check_forward);
}

/// Stride 1 only: the input gradient declines strided convolutions.
#[test]
fn simt_input_gradient_matches_the_definition() {
    let cases: Vec<Case> = CASES.iter().copied().filter(|c| c.stride == 1).collect();
    for_each_tiling(&cases, check_input_gradient);
}

#[test]
fn simt_weight_gradient_matches_the_definition() {
    // One tiling: the weight gradient has none to choose.
    let mut bad = Vec::new();
    for case in &CASES {
        match check_weight_gradient(case) {
            Ok(Ok(())) => {}
            Ok(Err(mismatch)) => bad.push(format!("{}: {mismatch}", case.name)),
            Err(err) => bad.push(format!("{}: declined: {err:?}", case.name)),
        }
    }
    assert!(bad.is_empty(), "failing cases:\n{}", bad.join("\n"));
}

/// What the kernels decline rather than compute wrongly.
#[test]
fn simt_declines_what_it_does_not_cover() {
    let case = CASES[0];
    let x = ramp(case.x_shape().iter().product(), 4);
    let w = ramp(case.w_shape().iter().product(), 3);
    let (_x, x_binding) = nhwc(&x, case.x_shape(), true);
    let (_w, w_binding) = nhwc(&w, case.w_shape(), true);
    let out = output(case.y_shape());
    let grouped = launch_simt(
        &client(),
        SimtTensors {
            input: x_binding.clone(),
            weight: w_binding.clone(),
            bias: None,
            out: out.clone().binding(),
        },
        case.args(),
        2,
        f32_type(),
        SimtTiling::NARROW,
    );
    assert!(matches!(grouped, Err(ConvSetupError::Groups(2))));

    // An output with NCHW strides is not one the kernel writes.
    let (_y, y_strided) = nhwc(
        &vec![0.0; case.y_shape().iter().product()],
        case.y_shape(),
        false,
    );
    let strided_out = launch_simt(
        &client(),
        SimtTensors {
            input: x_binding,
            weight: w_binding,
            bias: None,
            out: y_strided,
        },
        case.args(),
        1,
        f32_type(),
        SimtTiling::NARROW,
    );
    assert!(matches!(strided_out, Err(ConvSetupError::Unsupported(_))));
}
