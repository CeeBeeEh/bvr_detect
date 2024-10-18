mod model_config;
pub mod processing_type;
mod device_type;
mod bvr_image;
mod bvr_box;
mod bvr_detection;
mod config_ort;
mod yolo_types;
mod filesystem_access;
mod time_calc;
pub mod send_channels;
mod config_tch;

pub use config_ort::ConfigOrt;
pub use model_config::ModelConfig;
pub use processing_type::ProcessingType;
pub use device_type::DeviceType;

pub use bvr_image::BvrImage;
pub use bvr_detection::BvrDetection;

pub use crate::detection_runners::ort_detector::image_ops::ImageOps;
pub use crate::detection_runners::ort_detector::min_opt_max::MinOptMax;
pub use crate::detection_runners::ort_detector::xs::*;
pub use crate::detection_runners::ort_detector::y::Y;
pub use crate::detection_runners::ort_detector::dyn_conf::DynConf;
pub use crate::detection_runners::ort_detector::input_wrapper::X;

pub use yolo_types::*;
pub use filesystem_access::FsAccess;
pub use time_calc::TimeCalc;

pub(crate) const CROSS_MARK: &str = "❌";

