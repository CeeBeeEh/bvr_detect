pub mod inference_process;
pub(crate) mod ort_detector;
mod torch_detector;

pub use ort_detector::*;
pub use torch_detector::*;
