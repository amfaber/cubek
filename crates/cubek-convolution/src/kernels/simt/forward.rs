//! The SIMT convolution kernel, and the two launches that run it: the forward pass and the input
//! gradient of a stride-1 convolution.

use cubecl::{client::Client, ir::ElemType, prelude::*, zspace::Shape};

use super::{
    BK, PadBegin, PadBeginLaunch, SimtShape, SimtShapeLaunch, SimtTiling, check_common, check_len,
    contiguous_empty, is_contiguous,
    layout::{contiguous_nhwc, reorder_weight},
    sum_splits, vectors,
};
use crate::{components::ConvSetupError, launch::ConvolutionArgs};

/// Loads the input vector `k / 4` of the reduction (`k` a multiple of 4) for this unit's four
/// positions, zero outside the input or past `k_end`. Rows and columns are counted in the padded
/// input, `oy * sh + ky` and so on, so that no signed (or 64-bit) arithmetic is needed: WGSL has
/// no 64-bit integers.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_input<F: Float, N: Size>(
    x: &[Vector<F, N>],
    a_reg: &mut Array<Vector<F, N>>,
    base4: &Array<usize>,
    iy0: &Array<usize>,
    ix0: &Array<usize>,
    a_ok: &Array<bool>,
    k: usize,
    k_end: usize,
    s: &SimtShape,
    pad: &PadBegin,
) {
    let in_k = k < k_end;
    let kc = if in_k { k } else { 0usize };
    let tap = kc / s.c_in;
    let ci4 = (kc % s.c_in) / 4;
    let ky = (tap / s.kw) * s.dh;
    let kx = (tap % s.kw) * s.dw;
    let c_in4 = s.c_in / 4;
    #[unroll]
    for q in 0..4usize {
        let iy = iy0[q] + ky;
        let ix = ix0[q] + kx;
        let ok = in_k
            && a_ok[q]
            && iy >= pad.top
            && iy < pad.top + s.in_h
            && ix >= pad.left
            && ix < pad.left + s.in_w;
        // Clamped, so the read stays inside `x` whatever `ok` says.
        let at = if ok {
            base4[q] + ((iy - pad.top) * s.in_w + ix - pad.left) * c_in4 + ci4
        } else {
            0usize
        };
        let v = x[at];
        a_reg[q] = if ok {
            v
        } else {
            Vector::<F, N>::new(F::new(0.0))
        };
    }
}

/// `out[p, co] = sum over k of x[receptive field of p at k] * w[k, co]` (plus `bias[co]`), for a
/// contiguous NHWC `x` with `c_in % 4 == 0`, a contiguous `[kh, kw, c_in, c_out]` `w` and a
/// contiguous NHWC `out`; with a split, slice `CUBE_POS_Z` of `out` takes the partial sums of
/// reduction elements `CUBE_POS_Z * k_per ..+ k_per`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn conv_kernel<F: Float, N: Size>(
    x: &[Vector<F, N>],
    w: &[Vector<F, N>],
    bias: &[Vector<F, N>],
    out: &mut [Vector<F, N>],
    s: SimtShape,
    pad: PadBegin,
    c_out: usize,
    k_per: usize,
    #[comptime] bn: usize,
    #[comptime] bm: usize,
    #[comptime] has_bias: bool,
    #[define(F)] _dtype: ElemType,
) {
    // `bm` units: `bm / 8` position groups of 8 by 8 channel groups.
    let tnv = comptime!(bn / 32);
    let bn4 = comptime!(bn / 4);
    let bm4 = comptime!(bm / 4);
    let positions = s.batch * s.out_h * s.out_w;
    let k_begin = CUBE_POS_Z as usize * k_per;
    let k_total = s.kh * s.kw * s.c_in;
    let k_end = if k_begin + k_per < k_total {
        k_begin + k_per
    } else {
        k_total
    };
    let c_in4 = s.c_in / 4;
    let c_out4 = c_out / 4;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 8;
    let ty = tid / 8;
    let p0 = CUBE_POS_X as usize * bm;
    let c04 = CUBE_POS_Y as usize * bn4;

    let mut a_s = Shared::<[Vector<F, N>]>::new_slice(comptime!(BK * bm / 4));
    let mut b_s = Shared::<[Vector<F, N>]>::new_slice(comptime!(BK * bn / 4));

    // Input loads: reduction vector `kv` of the stage, for positions `pg * 4 ..+ 4`, each
    // decoded once here.
    let kv = tid % 4;
    let pg = tid / 4;
    let mut base4 = Array::<usize>::new(4usize);
    let mut iy0 = Array::<usize>::new(4usize);
    let mut ix0 = Array::<usize>::new(4usize);
    let mut a_ok = Array::<bool>::new(4usize);
    #[unroll]
    for q in 0..4usize {
        let p = p0 + pg * 4 + q;
        let ok = p < positions;
        let lp = if ok { p } else { positions - 1 };
        let ox = lp % s.out_w;
        let oy = (lp / s.out_w) % s.out_h;
        let img = lp / (s.out_w * s.out_h);
        base4[q] = img * s.in_h * s.in_w * c_in4;
        iy0[q] = oy * s.sh;
        ix0[q] = ox * s.sw;
        a_ok[q] = ok;
    }
    let b_per = comptime!(BK * bn / 4 / bm);

    let zero = Vector::<F, N>::new(F::new(0.0));
    let mut acc = Array::<Vector<F, N>>::new(comptime!(8 * tnv));
    #[unroll]
    for i in 0..comptime!(8 * tnv) {
        acc[i] = zero;
    }
    let mut a_reg = Array::<Vector<F, N>>::new(4usize);
    let mut b_reg = Array::<Vector<F, N>>::new(b_per);

    load_input::<F, N>(
        x,
        &mut a_reg,
        &base4,
        &iy0,
        &ix0,
        &a_ok,
        k_begin + kv * 4,
        k_end,
        &s,
        &pad,
    );
    #[unroll]
    for j in 0..b_per {
        let idx = tid + j * bm;
        let row = k_begin + idx / bn4;
        let col4 = idx % bn4;
        let ok = c04 + col4 < c_out4 && row < k_end;
        let at = if ok {
            row * c_out4 + c04 + col4
        } else {
            0usize
        };
        let v = w[at];
        b_reg[j] = if ok { v } else { zero };
    }

    let mut k0 = k_begin;
    while k0 < k_end {
        // Lane q of vector (k, pg) is position pg * 4 + q.
        #[unroll]
        for e in 0..4usize {
            let mut v = zero;
            #[unroll]
            for q in 0..4usize {
                v.insert(q, a_reg[q].extract(e));
            }
            a_s[(kv * 4 + e) * bm4 + pg] = v;
        }
        #[unroll]
        for j in 0..b_per {
            b_s[tid + j * bm] = b_reg[j];
        }
        sync_cube();

        let k1 = k0 + BK;
        if k1 < k_end {
            load_input::<F, N>(
                x,
                &mut a_reg,
                &base4,
                &iy0,
                &ix0,
                &a_ok,
                k1 + kv * 4,
                k_end,
                &s,
                &pad,
            );
            #[unroll]
            for j in 0..b_per {
                let idx = tid + j * bm;
                let row = k1 + idx / bn4;
                let col4 = idx % bn4;
                let ok = c04 + col4 < c_out4 && row < k_end;
                let at = if ok {
                    row * c_out4 + c04 + col4
                } else {
                    0usize
                };
                let v = w[at];
                b_reg[j] = if ok { v } else { zero };
            }
        }

        #[unroll]
        for kk in 0..BK {
            let a0 = a_s[kk * bm4 + ty * 2];
            let a1 = a_s[kk * bm4 + ty * 2 + 1];
            let mut b = Array::<Vector<F, N>>::new(tnv);
            #[unroll]
            for j in 0..tnv {
                b[j] = b_s[kk * bn4 + tx * tnv + j];
            }
            #[unroll]
            for i in 0..8usize {
                let ai = if comptime!(i < 4) {
                    a0.extract(i % 4)
                } else {
                    a1.extract(i % 4)
                };
                let av = Vector::<F, N>::new(ai);
                #[unroll]
                for j in 0..tnv {
                    acc[i * tnv + j] = fma(av, b[j], acc[i * tnv + j]);
                }
            }
        }
        sync_cube();
        k0 = k1;
    }

    let slice = CUBE_POS_Z as usize * positions * c_out4;
    #[unroll]
    for i in 0..8usize {
        let p = p0 + ty * 8 + i;
        if p < positions {
            #[unroll]
            for j in 0..tnv {
                let col4 = c04 + tx * tnv + j;
                if col4 < c_out4 {
                    let mut v = acc[i * tnv + j];
                    if comptime!(has_bias) {
                        // Once, in the first slice.
                        if CUBE_POS_Z == 0 {
                            v += bias[col4];
                        }
                    }
                    out[slice + p * c_out4 + col4] = v;
                }
            }
        }
    }
}

/// The bindings of [`launch_simt`].
pub struct SimtTensors {
    /// NHWC, any strides.
    pub input: TensorBinding,
    /// `[c_out, kh, kw, c_in]`, any strides.
    pub weight: TensorBinding,
    /// `[c_out]`, contiguous.
    pub bias: Option<TensorBinding>,
    /// NHWC, contiguous.
    pub out: TensorBinding,
}

/// The forward pass of a 2D convolution on the SIMT kernel; see the [module](super) for what it
/// declines. Input channels short of a multiple of 4 (an RGB image) are padded with zero
/// channels, in the copy the input takes anyway when it is not contiguous NHWC.
pub fn launch_simt(
    client: &Client,
    tensors: SimtTensors,
    args: ConvolutionArgs<2>,
    groups: usize,
    dtype: ElemType,
    tiling: SimtTiling,
) -> Result<(), ConvSetupError> {
    let SimtTensors {
        input,
        weight,
        bias,
        out,
    } = tensors;
    check_common(client, groups, dtype)?;
    let [batch, out_h, out_w, c_out] = dims(&out)?;
    let [in_batch, _, _, c_in] = dims(&input)?;
    let c_in4 = c_in.next_multiple_of(4);
    let [w_out, kh, kw, w_in] = dims(&weight)?;
    if in_batch != batch || w_out != c_out || w_in != c_in {
        return Err(ConvSetupError::Unsupported(
            "the input, weight and output shapes disagree",
        ));
    }
    check_out(&out, c_out)?;
    if let Some(bias) = &bias
        && !is_contiguous(bias)
    {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution takes a contiguous bias",
        ));
    }
    let k_total = kh * kw * c_in4;
    check_len(dims(&input)?[..3].iter().product::<usize>() * c_in4)?;
    check_len(k_total * c_out)?;
    tiling.check(batch * out_h * out_w, c_out, k_total)?;

    let input = contiguous_nhwc(client, input, c_in4, dtype);
    let weight = reorder_weight(client, weight, c_in4, false, dtype);
    launch(
        client,
        input,
        weight,
        bias,
        out,
        args.padding,
        args.stride,
        args.dilation,
        tiling,
        dtype,
    );
    Ok(())
}

/// The bindings of [`launch_simt_backward_data`].
pub struct SimtBackwardDataTensors {
    /// The output gradient, NHWC, any strides.
    pub out_grad: TensorBinding,
    /// The forward weight, `[c_out, kh, kw, c_in]`, any strides.
    pub weight: TensorBinding,
    /// The input gradient, NHWC, contiguous.
    pub in_grad: TensorBinding,
}

/// The input gradient of a stride-1 2D convolution with forward `args`: the convolution of the
/// output gradient by the flipped, channel-swapped window, padded by `dilation * (k - 1) -
/// padding`. Strided convolutions decline (inserting zeros would cost `stride^2` times the
/// work), as do channel counts that are not multiples of 4.
pub fn launch_simt_backward_data(
    client: &Client,
    tensors: SimtBackwardDataTensors,
    args: ConvolutionArgs<2>,
    groups: usize,
    dtype: ElemType,
    tiling: SimtTiling,
) -> Result<(), ConvSetupError> {
    let SimtBackwardDataTensors {
        out_grad,
        weight,
        in_grad,
    } = tensors;
    check_common(client, groups, dtype)?;
    if args.stride != [1, 1] {
        return Err(ConvSetupError::Unsupported(
            "the SIMT input gradient takes stride 1 only",
        ));
    }
    let [batch, out_h, out_w, c_out] = dims(&out_grad)?;
    let [in_batch, in_h, in_w, c_in] = dims(&in_grad)?;
    let [w_out, kh, kw, w_in] = dims(&weight)?;
    if in_batch != batch || w_out != c_out || w_in != c_in {
        return Err(ConvSetupError::Unsupported(
            "the output gradient, weight and input gradient shapes disagree",
        ));
    }
    // The roles swap: the output gradient's channels are the reduction's.
    if !c_out.is_multiple_of(4) {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution reduces over a multiple of 4 channels",
        ));
    }
    check_out(&in_grad, c_in)?;
    let mut pad_begin = [0; 2];
    for (axis, (out, k, in_size)) in [(out_h, kh, in_h), (out_w, kw, in_w)]
        .into_iter()
        .enumerate()
    {
        let reach = args.dilation[axis] * (k - 1);
        let invalid = ConvSetupError::Unsupported(
            "the SIMT input gradient needs padding below the window's reach",
        );
        pad_begin[axis] = reach.checked_sub(args.padding[axis]).ok_or(invalid)?;
        if in_size + reach < out + pad_begin[axis] {
            return Err(ConvSetupError::Unsupported(
                "the output gradient is larger than the input gradient allows",
            ));
        }
    }
    let k_total = kh * kw * c_out;
    check_len(batch * out_h * out_w * c_out)?;
    check_len(k_total * c_in)?;
    tiling.check(batch * in_h * in_w, c_in, k_total)?;

    let out_grad = contiguous_nhwc(client, out_grad, c_out, dtype);
    let weight = reorder_weight(client, weight, c_in, true, dtype);
    launch(
        client,
        out_grad,
        weight,
        None,
        in_grad,
        pad_begin,
        [1, 1],
        args.dilation,
        tiling,
        dtype,
    );
    Ok(())
}

/// The four dims of an NHWC (or OHWI) binding.
fn dims(binding: &TensorBinding) -> Result<[usize; 4], ConvSetupError> {
    match binding.shape.len() {
        4 => Ok([0, 1, 2, 3].map(|axis| binding.shape[axis])),
        _ => Err(ConvSetupError::Unsupported(
            "the SIMT convolution takes two spatial dimensions",
        )),
    }
}

/// The output: contiguous NHWC, in 4-wide vectors of channels.
fn check_out(out: &TensorBinding, channels: usize) -> Result<(), ConvSetupError> {
    if !channels.is_multiple_of(4) {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution writes a multiple of 4 channels",
        ));
    }
    if !is_contiguous(out) {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution writes a contiguous NHWC output",
        ));
    }
    check_len(out.shape.iter().product())
}

/// Runs the kernel on a contiguous NHWC `input`, `pad_begin` zero rows and columns before it,
/// and a contiguous `[kh, kw, c_in, c_out]` `weight`, into the contiguous NHWC `out`.
#[allow(clippy::too_many_arguments)]
fn launch(
    client: &Client,
    input: TensorBinding,
    weight: TensorBinding,
    bias: Option<TensorBinding>,
    out: TensorBinding,
    pad_begin: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    tiling: SimtTiling,
    dtype: ElemType,
) {
    let [batch, in_h, in_w, c_in] = [0, 1, 2, 3].map(|axis| input.shape[axis]);
    let [kh, kw, _, c_out] = [0, 1, 2, 3].map(|axis| weight.shape[axis]);
    let [_, out_h, out_w, _] = [0, 1, 2, 3].map(|axis| out.shape[axis]);
    let bn = SimtTiling::bn(c_out);
    let bm = tiling.bm;
    let positions = batch * out_h * out_w;
    let k_total = kh * kw * c_in;
    let splits = tiling.splits(positions, c_out, k_total);
    let k_per = k_total.div_ceil(BK).div_ceil(splits) * BK;
    let len = positions * c_out / 4;
    // A split writes its slices apart, and sums them into `out` after.
    let part = match splits {
        1 => out.clone(),
        _ => contiguous_empty(
            client,
            Shape::new([splits * batch, out_h, out_w, c_out]),
            dtype,
        ),
    };
    let w_len = kh * kw * c_in * c_out / 4;
    let bias_arg = match &bias {
        Some(bias) => vectors(bias, c_out / 4),
        // Never read; any buffer will do.
        None => vectors(&weight, w_len),
    };
    unsafe {
        conv_kernel::launch_unchecked(
            client,
            CubeCount::Static(
                positions.div_ceil(bm) as u32,
                c_out.div_ceil(bn) as u32,
                splits as u32,
            ),
            CubeDim::new_1d(bm as u32),
            4,
            vectors(&input, batch * in_h * in_w * c_in / 4),
            vectors(&weight, w_len),
            bias_arg,
            vectors(&part, splits * len),
            SimtShapeLaunch::new(
                batch,
                in_h,
                in_w,
                c_in,
                out_h,
                out_w,
                kh,
                kw,
                stride[0],
                stride[1],
                dilation[0],
                dilation[1],
            ),
            PadBeginLaunch::new(pad_begin[0], pad_begin[1]),
            c_out,
            k_per,
            bn,
            bm,
            bias.is_some(),
            dtype,
        )
    };
    if splits > 1 {
        sum_splits(client, &part, &out, len, splits, dtype);
    }
}
