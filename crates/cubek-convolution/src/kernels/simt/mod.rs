//! Implicit-gemm 2D convolution on plain GPU arithmetic ("SIMT"), NHWC, for devices with no
//! tensor-core tile for the element types: f32 on wgpu, whose Vulkan cooperative matrices take
//! only f16 and narrower inputs, and GPUs without tensor cores at all. There the accelerated
//! implicit-gemm routines report no tile size.
//!
//! The SGEMM shape: a cube of `bm` units (128 or 256, see [`SimtTiling`]) computes `bm` output
//! positions by `bn` output channels (32 or 64), walking the reduction (tap, input channel) `BK`
//! at a time. A stage of the input and of the weights goes through shared memory in 4-wide
//! vectors; the input is transposed in registers on its way there, so that a vector holds 4
//! positions of one reduction element. Each unit accumulates 8 positions by `bn / 32` vectors of
//! channels, and the next stage's global loads are issued before the current one is computed, so
//! that they overlap it. Where the tiles alone would leave the device idle, the reduction can be
//! split across cubes, their partial sums added after.
//!
//! Each unit decodes its output positions once, outside the reduction, rather than mapping every
//! load through a layout; the padding is implicit (a tap outside the input reads zero).
//!
//! Three launches share the kernels:
//! - [`launch_simt`]: the forward pass.
//! - [`launch_simt_backward_data`]: the input gradient of a stride-1 convolution, as the forward
//!   kernel run on the output gradient with the flipped, channel-swapped window.
//! - [`launch_simt_backward_weight`]: the weight gradient, the same gemm with the output positions
//!   as the reduction.
//!
//! Scope: two spatial dimensions, one group, 32-bit indexing; channel counts a multiple of 4
//! (input channels of the forward pass excepted: they are padded with zero channels). Everything
//! else declines with [`ConvSetupError::Unsupported`], as do CPU runtimes.

mod backward_weight;
mod forward;
mod layout;

pub use backward_weight::{SimtBackwardWeightTensors, launch_simt_backward_weight};
pub use forward::{SimtBackwardDataTensors, SimtTensors, launch_simt, launch_simt_backward_data};

use cubecl::{
    client::Client,
    ir::{ElemType, FloatKind},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};

use crate::components::ConvSetupError;

/// Reduction elements per stage.
const BK: usize = 16;

/// The shape of the gemm a kernel runs, unpadded: the input `[batch, in_h, in_w, c_in]`, the
/// output positions `[batch, out_h, out_w]`, the window, its stride and dilation.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
struct SimtShape {
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_in: usize,
    out_h: usize,
    out_w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
}

/// Zero rows and columns before the input.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
struct PadBegin {
    top: usize,
    left: usize,
}

/// How [`launch_simt`] and [`launch_simt_backward_data`] tile a convolution. None wins
/// everywhere, so a caller with an autotuner tries each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimtTiling {
    /// Output positions per cube, and units per cube: 128 or 256.
    bm: usize,
    /// Whether to split the reduction across cubes when the tiles alone would not fill the
    /// device. A split tiling declines where it would not split, being the unsplit one there.
    split: bool,
}

impl SimtTiling {
    /// 128 positions per cube.
    pub const NARROW: Self = Self {
        bm: 128,
        split: false,
    };
    /// [`NARROW`](Self::NARROW), the reduction split across cubes where the tiles alone would
    /// leave the device idle: deep layers with few output positions.
    pub const NARROW_SPLIT: Self = Self {
        bm: 128,
        split: true,
    };
    /// Twice the positions per cube, each stage of the weights serving twice as many; needs 64
    /// output channels or more.
    pub const WIDE: Self = Self {
        bm: 256,
        split: false,
    };

    fn bn(c_out: usize) -> usize {
        if c_out >= 64 { 64 } else { 32 }
    }

    /// Slices of the reduction: enough cubes for about 256 of them, with no slice shorter than
    /// 8 stages. (No count of streaming multiprocessors is available on every runtime.)
    fn splits(self, positions: usize, c_out: usize, k_total: usize) -> usize {
        if !self.split {
            return 1;
        }
        let tiles = positions.div_ceil(self.bm) * c_out.div_ceil(Self::bn(c_out));
        let stages = k_total.div_ceil(BK);
        let splits = 256usize.div_ceil(tiles).clamp(1, (stages / 8).max(1));
        k_total.div_ceil(stages.div_ceil(splits) * BK)
    }

    fn check(self, positions: usize, c_out: usize, k_total: usize) -> Result<(), ConvSetupError> {
        if self.bm == 256 && c_out < 64 {
            return Err(ConvSetupError::Unsupported(
                "the wide SIMT tiling needs 64 output channels or more",
            ));
        }
        if self.split && self.splits(positions, c_out, k_total) == 1 {
            return Err(ConvSetupError::Unsupported(
                "the split SIMT tiling would not split this reduction",
            ));
        }
        Ok(())
    }
}

/// What every SIMT launch declines: CPU runtimes (whose pools run these kernels poorly), grouped
/// convolutions, and element types other than floats.
fn check_common(client: &Client, groups: usize, dtype: ElemType) -> Result<(), ConvSetupError> {
    if client.properties().hardware.num_cpu_cores.is_some() {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution targets GPUs, not CPU runtimes",
        ));
    }
    if groups != 1 {
        return Err(ConvSetupError::Groups(groups));
    }
    if !matches!(
        dtype,
        ElemType::Float(FloatKind::F32 | FloatKind::F16 | FloatKind::BF16)
    ) {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution takes f32, f16 or bf16",
        ));
    }
    Ok(())
}

/// The kernels index with 32-bit offsets.
fn check_len(elems: usize) -> Result<(), ConvSetupError> {
    match elems < u32::MAX as usize {
        true => Ok(()),
        false => Err(ConvSetupError::Unsupported(
            "the SIMT convolution indexes with 32 bits",
        )),
    }
}

/// Whether `binding` is contiguous in the order of its shape, row-major.
fn is_contiguous(binding: &TensorBinding) -> bool {
    let mut expected = 1;
    for (dim, stride) in binding.shape.iter().zip(binding.strides.iter()).rev() {
        if *dim != 1 && *stride != expected {
            return false;
        }
        expected *= *dim;
    }
    true
}

/// `binding` as a buffer of `len` 4-wide vectors.
fn vectors(binding: &TensorBinding, len: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts_binding(binding.handle.clone(), len) }
}

/// A new contiguous tensor, without the row padding an optimized layout may add.
fn contiguous_empty(client: &Client, shape: Shape, dtype: ElemType) -> TensorBinding {
    let bytes = shape.iter().product::<usize>() * dtype.size();
    TensorHandle::new_contiguous(shape, client.empty(bytes), dtype).binding()
}

/// `out[i] = sum over splits of part[split][i]`, in vectors.
#[cube(launch_unchecked)]
fn sum_splits_kernel<F: Float, N: Size>(
    part: &[Vector<F, N>],
    out: &mut [Vector<F, N>],
    len: usize,
    splits: usize,
    #[define(F)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS;
    if i < len {
        let mut acc = part[i];
        for split in 1..splits {
            acc += part[split * len + i];
        }
        out[i] = acc;
    }
}

/// Adds the `splits` slices of `part`, `len` vectors each, into `out`.
fn sum_splits(
    client: &Client,
    part: &TensorBinding,
    out: &TensorBinding,
    len: usize,
    splits: usize,
    dtype: ElemType,
) {
    unsafe {
        sum_splits_kernel::launch_unchecked(
            client,
            CubeCount::Static(len.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            4,
            vectors(part, splits * len),
            vectors(out, len),
            len,
            splits,
            dtype,
        )
    };
}
