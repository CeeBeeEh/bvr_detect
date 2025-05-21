mod utils;
mod detectors;
pub mod data;
pub mod detection_runners;
pub mod common;

use std::{time::Instant};
use image::DynamicImage;
use log;
use ort::Error;
use crate::common::{BvrDetection, BvrImage, BvrOrtYOLO, ModelConfig};
use crate::data::ConfigOrt;
use crate::detectors::{detector_onnx};
use crate::detection_runners::inference_process::InferenceProcess;

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub fn init_detector(model_details: &ModelConfig, is_test: bool) -> anyhow::Result<BvrOrtYOLO> {
    println!("===========\ninit_detector\n===========");
    let ort_options = ConfigOrt::new()
        .with_model(&model_details.weights_path)?
        .with_ort_lib_path(&model_details.ort_lib_path)?
        //.with_batch_size(2)
        .with_yolo_version(model_details.model_version)
        .with_device(model_details.inference_device)
        .with_trt_fp16(false)
        .with_ixx(0, 0, (1, 1 as _, 4).into())
        .with_ixx(0, 2, (320, 544, 1200).into())
        .with_ixx(0, 3, (320, 960, 1200).into())
        .with_confs(&[model_details.get_threshold()])
        .with_nc(29)
        .with_profile(false);

    log::info!("Initializing ORT session with ({}) execution provider", model_details.inference_device.to_string());
    let mut yolo = BvrOrtYOLO::new(ort_options)?;
    yolo.run(&[DynamicImage::new_rgb8(960, 960)], &[vec![0.5]])?;
    Ok(yolo)
}

pub fn run_detection(yolo: &mut BvrOrtYOLO, bvr_image: BvrImage, model_details: &ModelConfig) -> anyhow::Result<Vec<BvrDetection>> {
    let now = Instant::now();

    let detections = detector_onnx(false, yolo, bvr_image, &model_details)?;

    println!("Processing time: {:?}", now.elapsed());

    Ok(detections)
}


