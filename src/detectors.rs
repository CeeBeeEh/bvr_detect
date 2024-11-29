use std::fs;
use std::path::Path;
use std::time::Instant;
use image::{DynamicImage, EncodableLayout};
/*use pyo3::prelude::*;
use pyo3::Python;
use pyo3::types::PyByteArray;*/

use crate::data::{BvrDetection, BvrImage, ModelConfig, ConfigOrt, YoloVersion};
use crate::data::send_channels::DetectionState;
use crate::detection_runners::inference_process::InferenceProcess;
use crate::detection_runners::OrtYOLO;

pub fn detector_onnx(is_test: bool, detection_state: DetectionState, model_details: ModelConfig) -> anyhow::Result<()> {
    let threshold = model_details.conf_threshold;

    let ort_options = ConfigOrt::new()
        .with_model(&model_details.weights_path)?
        .with_ort_lib_path(&model_details.ort_lib_path)?
        //.with_batch_size(2)
        .with_yolo_version(YoloVersion::V11)
        .with_device(model_details.device_type)
        .with_trt_fp16(false)
        .with_ixx(0, 0, (1, 1 as _, 4).into())
        .with_ixx(0, 2, (640, 640, 640).into())
        .with_ixx(0, 3, (640, 640, 640).into())
        .with_confs(&[threshold])
        .with_nc(80)
        .with_profile(false);

    log::info!("Initializing ORT session with ({}) execution provider", model_details.device_type.as_str());
    let mut yolo = OrtYOLO::new(ort_options)?;

    // Dry run to init model, we want to fail here if there's an issue
    yolo.run(&[DynamicImage::new_rgb8(model_details.width, model_details.height)]).expect("Dry run failed");

    loop {
        // MESSAGE LOOP STARTS HERE
        let bvr_image_box = match detection_state.opt_rx.recv() {
            Ok(msg) => msg,
            Err(err) => {
                log::error!("bvr_detect: Failed to receive image: {}", err);
                continue;
            }
        };
        let detect_time = Instant::now();

        let bvr_image: BvrImage = *bvr_image_box;

        let mut detections: Vec<BvrDetection> = vec![];

        // A ratio of 1.777~ is 16/9
        if model_details.split_wide_input && bvr_image.get_ratio() > 1.78 {
            let crop_w = (bvr_image.img_width / 2) as u32;
            let img_left = bvr_image.image.crop_imm(0, 0, crop_w, bvr_image.img_height as u32);
            let img_right = bvr_image.image.crop_imm(crop_w, 0, bvr_image.img_width as u32, bvr_image.img_height as u32);

            let ys = yolo.forward(&[img_left], false)?;
            let ys_right = yolo.forward(&[img_right], false)?;

            match ys[0].detections() {
                None => {}
                _ => {
                    detections = ys[0].detections().expect("Error parsing results vector!").to_vec();

                    match ys_right[0].detections() {
                        None => {}
                        _ => {
                            for detection in ys_right[0].detections().unwrap() {
                                let mut bvr_det = detection.clone();
                                bvr_det.bbox.x1 += crop_w as f32;
                                bvr_det.bbox.x2 += crop_w as f32;
                                detections.push(bvr_det);
                            }
                        }
                    }
                }
            }
        }
        else {
            let ys = yolo.forward(&[bvr_image.image], false)?;

            match ys[0].detections() {
                None => {}
                _ => {
                    detections = ys[0].detections().expect("Error parsing results vector!").to_vec();
                }
            }
        }

        // remove any detection that's not in the wanted_labels vector
        if let Some(wanted) = bvr_image.wanted_labels {
            detections.retain(|detection| wanted.contains(&(detection.class_id as u16)));
        }
        
        detection_state.det_tx.send(Box::from(detections))?;

        println!("Loop time: {:?}", detect_time.elapsed());

        continue
    }
}

// /// The python detector is still a WIP
/*pub fn detector_python(is_test: bool, detection_state: DetectionState, model_details: ModelConfig) -> anyhow::Result<()> {
    let threshold = model_details.conf_threshold;

    log::info!("Initializing ORT session with ({}) execution provider", model_details.device_type.as_str());
    /*
    -------------- DO NOT USE --------------
    THIS IS NOT READY AND WILL CAUSE A PANIC
    */

    let python_file = std::fs::read_to_string("/mnt/4TB/Development/Bvr-Project/bvr_detector_lib_python/detector.py")?;

    let model_path = "/mnt/4TB/Development/Bvr-Project/bvr_detector_lib_python/yolov9-t.pt";

    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let sys = py.import_bound("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        //path.call_method1("insert", (0, model_path )).unwrap();
        path.call_method1("append", ("/mnt/4TB/Development/Bvr-Project/bvr_detector_lib_python/venv3/lib/python3.10/site-packages",)).unwrap();  // append venv path
        path.call_method1("append", ("/mnt/4TB/Development/Bvr-Project/bvr_detector_lib_python",)).unwrap();

        for entry in fs::read_dir("/mnt/4TB/Development/Bvr-Project/bvr_detector_lib_python").unwrap() {
            let entry = entry.unwrap();
            let entry_path = entry.path();

            // Check if the entry is a directory
            if entry_path.is_dir() {
                path.call_method1("append", (entry_path,)).unwrap();
            }
        }

        let p_module = PyModule::from_code_bound(py, &python_file, "detector.py", "detector").unwrap();

        let init_fun: Py<PyAny> = p_module.getattr("init_detector").unwrap().unbind();
        let detect_fun: Py<PyAny> = p_module.getattr("run_detector").unwrap().unbind();

        let device = "cpu"; // Or "cuda" depending on your setup
        let model_size = 640;
        let conf_thresh = 0.5;
        let iou_thresh = 0.4;

        let args = (device, model_path, model_size, conf_thresh, iou_thresh);

        match init_fun.call1(py, args) {
            Ok(_) => { println!("worked")}
            Err(e) => { println!("{:?}", e) }
        }

        let img_path = "/mnt/4TB/Development/Bvr-Project/bvr_detect/tests/8_people.jpg";

        let dyn_image = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join(img_path)).unwrap();

        let rgb8 = dyn_image.clone().to_rgb8();
        let img_bytes = rgb8.as_bytes();

        let py_bytes = PyByteArray::new_bound(py,&img_bytes);
        let py_image = py_bytes.as_any();

        let args = (py_image, 0, 0);

        let results = detect_fun.call1(py, args);

        println!("{:?}", results);
    });

    Ok(())
}*/
