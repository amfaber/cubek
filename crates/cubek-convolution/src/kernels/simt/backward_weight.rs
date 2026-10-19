//! The weight gradient of a 2D convolution on plain GPU arithmetic: the forward kernel's gemm
//! with (tap, input channel) as its rows and the output positions as its reduction. Nothing the
//! size of the input times the window is materialized, unlike an im2col.

use cubecl::{client::Client, ir::ElemType, prelude::*, zspace::Shape};

use super::{
    PadBegin, PadBeginLaunch, SimtShape, SimtShapeLaunch, SimtTiling, check_common, check_len,
    contiguous_empty, layout::contiguous_nhwc, sum_splits, vectors,
};
use crate::{components::ConvSetupError, launch::ConvolutionArgs};

/// Reduction rows (tap, input channel) per cube.
const WM: usize = 128;
/// Output positions per stage.
const WP: usize = 16;
/// Units per cube: 16 row groups of 8 by 8 channel groups.
const UNITS: usize = 128;

/// `dw[(tap, ci), co] = sum over positions p of x[receptive field of p at tap, ci] * dy[p, co]`.
/// A stage holds 16 positions of 128 rows of the input, 4-wide along the input channels, and 16
/// positions of the output gradient; the grid's third axis splits the positions, each split
/// writing its partial sums to `dw_part[split]`. The padding is implicit. Needs `c_in % 4 == 0`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn wgrad_kernel<F: Float, N: Size>(
    x: &[Vector<F, N>],
    dy: &[Vector<F, N>],
    dw_part: &mut [Vector<F, N>],
    s: SimtShape,
    pad: PadBegin,
    c_out: usize,
    per_split: usize,
    #[comptime] bn: usize,
    #[define(F)] _dtype: ElemType,
) {
    let tnv = comptime!(bn / 32);
    let bn4 = comptime!(bn / 4);
    let wm4 = comptime!(WM / 4);
    let positions = s.batch * s.out_h * s.out_w;
    let rows = s.kh * s.kw * s.c_in;
    let c_in4 = s.c_in / 4;
    let c_out4 = c_out / 4;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 8;
    let ty = tid / 8;
    let r0 = CUBE_POS_X as usize * WM;
    let c04 = CUBE_POS_Y as usize * bn4;
    let split = CUBE_POS_Z as usize;
    let q_begin = split * per_split;
    let q_end = if q_begin + per_split < positions {
        q_begin + per_split
    } else {
        positions
    };

    let mut a_s = Shared::<[Vector<F, N>]>::new_slice(comptime!(WP * WM / 4));
    let mut b_s = Shared::<[Vector<F, N>]>::new_slice(comptime!(WP * bn / 4));

    // Input loads: row vector `rv` (rows `r0 + rv * 4 ..+ 4`, within one tap), positions
    // `pg * 4 ..+ 4` of the stage. The tap and channels are this unit's for the whole reduction;
    // rows and columns are counted in the padded input, unsigned.
    let rv = tid % 32;
    let pg = tid / 32;
    let row = r0 + rv * 4;
    let row_ok = row < rows;
    let tap = row / s.c_in;
    let ky = (tap / s.kw) * s.dh;
    let kx = (tap % s.kw) * s.dw;
    let ci4 = (row % s.c_in) / 4;
    let b_per = comptime!(WP * bn / 4 / UNITS);

    let zero = Vector::<F, N>::new(F::new(0.0));
    let mut acc = Array::<Vector<F, N>>::new(comptime!(8 * tnv));
    #[unroll]
    for i in 0..comptime!(8 * tnv) {
        acc[i] = zero;
    }

    let mut q0 = q_begin;
    while q0 < q_end {
        #[unroll]
        for e in 0..4usize {
            let q = q0 + pg * 4 + e;
            let lq = if q < q_end { q } else { q_begin };
            let ox = lq % s.out_w;
            let oy = (lq / s.out_w) % s.out_h;
            let img = lq / (s.out_w * s.out_h);
            let iy = oy * s.sh + ky;
            let ix = ox * s.sw + kx;
            let ok = row_ok
                && q < q_end
                && iy >= pad.top
                && iy < pad.top + s.in_h
                && ix >= pad.left
                && ix < pad.left + s.in_w;
            // Clamped, so the read stays inside `x` whatever `ok` says.
            let at = if ok {
                ((img * s.in_h + iy - pad.top) * s.in_w + ix - pad.left) * c_in4 + ci4
            } else {
                0usize
            };
            let v = x[at];
            a_s[(pg * 4 + e) * wm4 + rv] = if ok { v } else { zero };
        }
        #[unroll]
        for j in 0..b_per {
            let idx = tid + j * UNITS;
            let pp = idx / bn4;
            let col4 = idx % bn4;
            let q = q0 + pp;
            let ok = q < q_end && c04 + col4 < c_out4;
            let at = if ok { q * c_out4 + c04 + col4 } else { 0usize };
            let v = dy[at];
            b_s[idx] = if ok { v } else { zero };
        }
        sync_cube();

        #[unroll]
        for pp in 0..WP {
            let a0 = a_s[pp * wm4 + ty * 2];
            let a1 = a_s[pp * wm4 + ty * 2 + 1];
            let mut b = Array::<Vector<F, N>>::new(tnv);
            #[unroll]
            for j in 0..tnv {
                b[j] = b_s[pp * bn4 + tx * tnv + j];
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
        q0 += WP;
    }

    let part = split * rows * c_out4;
    #[unroll]
    for i in 0..8usize {
        let r = r0 + ty * 8 + i;
        if r < rows {
            #[unroll]
            for j in 0..tnv {
                let col4 = c04 + tx * tnv + j;
                if col4 < c_out4 {
                    dw_part[part + r * c_out4 + col4] = acc[i * tnv + j];
                }
            }
        }
    }
}

/// The bindings of [`launch_simt_backward_weight`].
pub struct SimtBackwardWeightTensors {
    /// The forward input, NHWC, any strides.
    pub input: TensorBinding,
    /// The output gradient, NHWC, any strides.
    pub out_grad: TensorBinding,
    /// The weight gradient, of shape `[c_out, kh, kw, c_in]` and laid out in memory as a
    /// contiguous `[kh, kw, c_in, c_out]` (strides `[1, kw * c_in * c_out, c_in * c_out,
    /// c_out]`): the layout the kernel writes, so that no copy follows it.
    pub weight_grad: TensorBinding,
}

/// The weight gradient of a 2D convolution with forward `args`; see the [module](super) for what
/// it declines. Input and output channels must be multiples of 4.
pub fn launch_simt_backward_weight(
    client: &Client,
    tensors: SimtBackwardWeightTensors,
    args: ConvolutionArgs<2>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    let SimtBackwardWeightTensors {
        input,
        out_grad,
        weight_grad,
    } = tensors;
    check_common(client, groups, dtype)?;
    if [&input, &out_grad, &weight_grad]
        .iter()
        .any(|binding| binding.shape.len() != 4)
    {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution takes two spatial dimensions",
        ));
    }
    let [batch, in_h, in_w, c_in] = [0, 1, 2, 3].map(|axis| input.shape[axis]);
    let [_, out_h, out_w, c_out] = [0, 1, 2, 3].map(|axis| out_grad.shape[axis]);
    let [w_out, kh, kw, w_in] = [0, 1, 2, 3].map(|axis| weight_grad.shape[axis]);
    if out_grad.shape[0] != batch || w_out != c_out || w_in != c_in {
        return Err(ConvSetupError::Unsupported(
            "the input, output gradient and weight gradient shapes disagree",
        ));
    }
    if !c_in.is_multiple_of(4) || !c_out.is_multiple_of(4) {
        return Err(ConvSetupError::Unsupported(
            "the SIMT weight gradient takes multiples of 4 channels",
        ));
    }
    let expected = [1, kw * c_in * c_out, c_in * c_out, c_out];
    if weight_grad.strides.iter().copied().ne(expected) {
        return Err(ConvSetupError::Unsupported(
            "the SIMT weight gradient writes a contiguous [kh, kw, c_in, c_out]",
        ));
    }
    let rows = kh * kw * c_in;
    let positions = batch * out_h * out_w;
    check_len(batch * in_h * in_w * c_in)?;
    check_len(positions * c_out)?;

    let bn = SimtTiling::bn(c_out);
    let tiles = rows.div_ceil(WM) * c_out.div_ceil(bn);
    // Enough cubes to fill the device several times over, splitting the positions to get them,
    // with no split shorter than 16 stages.
    let splits = 512usize
        .div_ceil(tiles)
        .clamp(1, positions.div_ceil(WP * 16).max(1));
    let per_split = positions.div_ceil(splits).div_ceil(WP) * WP;
    let splits = positions.div_ceil(per_split);
    let len = rows * c_out / 4;
    check_len(splits * len * 4)?;

    let input = contiguous_nhwc(client, input, c_in, dtype);
    let out_grad = contiguous_nhwc(client, out_grad, c_out, dtype);
    let dw_part = match splits {
        1 => weight_grad.clone(),
        _ => contiguous_empty(client, Shape::new([splits * kh, kw, c_in, c_out]), dtype),
    };
    unsafe {
        wgrad_kernel::launch_unchecked(
            client,
            CubeCount::Static(
                rows.div_ceil(WM) as u32,
                c_out.div_ceil(bn) as u32,
                splits as u32,
            ),
            CubeDim::new_1d(UNITS as u32),
            4,
            vectors(&input, batch * in_h * in_w * c_in / 4),
            vectors(&out_grad, positions * c_out / 4),
            vectors(&dw_part, splits * len),
            SimtShapeLaunch::new(
                batch,
                in_h,
                in_w,
                c_in,
                out_h,
                out_w,
                kh,
                kw,
                args.stride[0],
                args.stride[1],
                args.dilation[0],
                args.dilation[1],
            ),
            PadBeginLaunch::new(args.padding[0], args.padding[1]),
            c_out,
            per_split,
            bn,
            dtype,
        )
    };
    if splits > 1 {
        sum_splits(client, &dw_part, &weight_grad, len, splits, dtype);
    }
    Ok(())
}
