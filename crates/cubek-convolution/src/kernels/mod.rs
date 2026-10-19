/// Kernels for data gradients
pub mod backward_data;
/// Kernels for weight gradients
pub mod backward_weight;
/// Kernels for forward convolution
pub mod forward;

/// Explicit convolution-window lowering
pub mod im2col;
/// Implicit-gemm convolution on plain GPU arithmetic, for devices without tensor cores for the
/// element types: forward, input gradient and weight gradient
pub mod simt;
