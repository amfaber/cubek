//! The layouts the SIMT kernels read, from whatever strides the caller has: the input as a
//! contiguous NHWC tensor, the weight as a contiguous `[kh, kw, c_in, c_out]` (or, for the input
//! gradient, `[kh, kw, c_out, c_in]` with the window flipped). One pass each, coalesced on the
//! GPU.

use cubecl::{client::Client, ir::ElemType, prelude::*, zspace::Shape};

use super::{contiguous_empty, is_contiguous};

/// `x` (NHWC, any strides) as a contiguous NHWC tensor of `channels >= c` channels, the ones
/// past its own zero. A cube per output row `(n, y)`, its units writing the row in order: the
/// reads gather across channels when `x` is NCHW in memory, but stay within one input row, which
/// the caches hold.
#[cube(launch_unchecked)]
fn nhwc_kernel<F: Float>(
    x: &Tensor<F>,
    out: &mut [F],
    channels: usize,
    #[define(F)] _dtype: ElemType,
) {
    let (h, w, c) = (x.shape(1), x.shape(2), x.shape(3));
    let row = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    if row >= x.shape(0) * h {
        terminate!();
    }
    let in_row = (row / h) * x.stride(0) + (row % h) * x.stride(1);
    let row_len = w * channels;
    let at = row * row_len;
    let mut i = UNIT_POS_X as usize;
    while i < row_len {
        let ci = i % channels;
        let mut value = F::new(0.0);
        if ci < c {
            value = x[in_row + (i / channels) * x.stride(2) + ci * x.stride(3)];
        }
        out[at + i] = value;
        i += CUBE_DIM_X as usize;
    }
}

/// `x` (an NHWC binding, any strides) as a contiguous NHWC tensor of `channels` channels:
/// itself when it already is one.
pub(super) fn contiguous_nhwc(
    client: &Client,
    x: TensorBinding,
    channels: usize,
    dtype: ElemType,
) -> TensorBinding {
    let [batch, h, w, c] = [0, 1, 2, 3].map(|axis| x.shape[axis]);
    if c == channels && is_contiguous(&x) {
        return x;
    }
    let out = contiguous_empty(client, Shape::new([batch, h, w, channels]), dtype);
    let rows = batch * h;
    let len = rows * w * channels;
    unsafe {
        nhwc_kernel::launch_unchecked(
            client,
            CubeCount::Static(
                rows.min(u16::MAX as usize) as u32,
                rows.div_ceil(u16::MAX as usize) as u32,
                1,
            ),
            CubeDim::new_1d(256),
            x.into_tensor_arg(),
            BufferArg::from_raw_parts_binding(out.handle.clone(), len),
            channels,
            dtype,
        )
    };
    out
}

/// A cube per 32 x 32 tile of (output channel, input channel) of one tap, staged through shared
/// memory so that both the reads (along the input channels) and the writes (along whichever
/// channels the output row runs) are coalesced. Input channels from `c_in` to `out_c_in` are
/// zero.
#[cube(launch_unchecked)]
fn reorder_weight_kernel<F: Float>(
    w: &Tensor<F>,
    out: &mut [F],
    out_c_in: usize,
    #[comptime] for_input_gradient: bool,
    #[define(F)] _dtype: ElemType,
) {
    let (c_out, kh, kw, c_in) = (w.shape(0), w.shape(1), w.shape(2), w.shape(3));
    let tap = CUBE_POS_Z as usize;
    let (ky, kx) = if comptime!(for_input_gradient) {
        (kh - 1 - tap / kw, kw - 1 - tap % kw)
    } else {
        (tap / kw, tap % kw)
    };
    let tap_at = ky * w.stride(1) + kx * w.stride(2);
    let ci0 = CUBE_POS_X as usize * 32;
    let co0 = CUBE_POS_Y as usize * 32;
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    // Row: an output channel; column: an input channel. One padding column against bank
    // conflicts on the transposed read.
    let mut tile = Shared::<[F]>::new_slice(32usize * 33);
    #[unroll]
    for j in 0..4usize {
        let co = co0 + ty + 8 * j;
        let ci = ci0 + tx;
        let mut value = F::new(0.0);
        if co < c_out && ci < c_in {
            value = w[tap_at + co * w.stride(0) + ci * w.stride(3)];
        }
        tile[(ty + 8 * j) * 33 + tx] = value;
    }
    sync_cube();
    #[unroll]
    for j in 0..4usize {
        if comptime!(for_input_gradient) {
            // `[tap, c_out, c_in]`: along the input channels, as read.
            let co = co0 + ty + 8 * j;
            let ci = ci0 + tx;
            if co < c_out && ci < out_c_in {
                out[(tap * c_out + co) * out_c_in + ci] = tile[(ty + 8 * j) * 33 + tx];
            }
        } else {
            // `[tap, c_in, c_out]`: along the output channels, transposed.
            let ci = ci0 + ty + 8 * j;
            let co = co0 + tx;
            if co < c_out && ci < out_c_in {
                out[(tap * out_c_in + ci) * c_out + co] = tile[tx * 33 + ty + 8 * j];
            }
        }
    }
}

/// The weight `[c_out, kh, kw, c_in]` (any strides) as a contiguous `[kh, kw, out_c_in, c_out]`,
/// or, with `for_input_gradient`, as `[kh, kw, c_out, out_c_in]` with the window flipped: the
/// weight of the convolution that gives the input gradient. Input channels past `c_in` are zero.
pub(super) fn reorder_weight(
    client: &Client,
    weight: TensorBinding,
    out_c_in: usize,
    for_input_gradient: bool,
    dtype: ElemType,
) -> TensorBinding {
    let [c_out, kh, kw, _] = [0, 1, 2, 3].map(|axis| weight.shape[axis]);
    let shape = match for_input_gradient {
        false => Shape::new([kh, kw, out_c_in, c_out]),
        true => Shape::new([kh, kw, c_out, out_c_in]),
    };
    let len = shape.iter().product();
    let out = contiguous_empty(client, shape, dtype);
    unsafe {
        reorder_weight_kernel::launch_unchecked(
            client,
            CubeCount::Static(
                out_c_in.div_ceil(32) as u32,
                c_out.div_ceil(32) as u32,
                (kh * kw) as u32,
            ),
            CubeDim::new_2d(32, 8),
            weight.into_tensor_arg(),
            BufferArg::from_raw_parts_binding(out.handle.clone(), len),
            out_c_in,
            for_input_gradient,
            dtype,
        )
    };
    out
}
