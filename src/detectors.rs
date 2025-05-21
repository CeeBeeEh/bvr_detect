use std::time::Instant;
use crate::common::{BvrDetection, BvrImage, ModelConfig, BvrOrtYOLO};
/*use pyo3::prelude::*;
use pyo3::Python;
use pyo3::types::PyByteArray;*/

use crate::detection_runners::inference_process::InferenceProcess;

pub fn detector_onnx(is_test: bool, yolo: &mut BvrOrtYOLO, bvr_image: BvrImage, model_details: &ModelConfig) -> anyhow::Result<Vec<BvrDetection>> {
    let detect_time = Instant::now();

    let mut detections: Vec<BvrDetection> = vec![];
    
    let threshold = bvr_image.threshold;

    // A ratio of 1.777~ is 16/9
    if model_details.split_wide_input && bvr_image.get_ratio() > 1.78 {
        let crop_w = bvr_image.get_img_width() - &bvr_image.img_height;
        //let crop_w = (bvr_image.img_width() / 2) as u32;
        let img_left = bvr_image.image.crop_imm(0, 0, bvr_image.get_img_height(), bvr_image.get_img_height());
        let img_right = bvr_image.image.crop_imm(crop_w, 0, bvr_image.get_img_width(), bvr_image.get_img_height());

        let ys = yolo.forward(&[img_left], &[vec![threshold]], false)?;
        let ys_right = yolo.forward(&[img_right], &[vec![threshold]], false)?;

        match ys[0].detections() {
            None => {}
            _ => {
                detections = ys[0].detections().expect("Error parsing results vector!").to_vec();

                match ys_right[0].detections() {
                    None => {}
                    _ => {
                        'right_loop: for detection in ys_right[0].detections().unwrap() {
                            let mut bvr_det = detection.clone();
                            bvr_det.bbox.x1 += crop_w as f32;
                            bvr_det.bbox.x2 += crop_w as f32;

                            for left_detection in ys[0].detections().unwrap() {
                                if &left_detection.class_id == &bvr_det.class_id
                                && left_detection.bbox.intersect(&bvr_det.bbox) > 0.95 {
                                   continue 'right_loop;
                                }
                            }

                            detections.push(bvr_det);
                        }
                    }
                }
            }
        }
    }
    else {
        let ys = yolo.forward(&[bvr_image.clone_image()], &[vec![threshold]], false)?;

        match ys[0].detections() {
            None => {}
            _ => {
                detections = ys[0].detections().expect("Error parsing results vector!").to_vec();
            }
        }
    }

    // remove any detection that's not in the wanted_labels vector
    if let Some(wanted) = bvr_image.wanted_labels {
        if !wanted.is_empty() {
            detections.retain(|detection| wanted.contains(&detection.get_class_id()));
        }
    }

    Ok(detections)
}

// /// The python detector is still a WIP
/*pub fn detector_python(is_test: bool, detection_state: DetectionState, model_details: ModelConfig) -> anyhow::Result<()> {
    let threshold = model_details.conf_threshold;

    log::info!("Initializing ORT session with ({}) execution provider", model_details.device_type.as_str());
    /*
    -------------- DO NOT USE --------------
    THIS IS NOT READY AND WILL CAUSE A PANIC
    */

    let python_file = std::fs::read_to_string("../bvr_detector_lib_python/detector.py")?;

    let model_path = "../bvr_detector_lib_python/yolov9-t.pt";

    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let sys = py.import_bound("sys").unwrap();
        let path = sys.getattr("path").unwrap();
        //path.call_method1("insert", (0, model_path )).unwrap();
        path.call_method1("append", ("../bvr_detector_lib_python/venv3/lib/python3.10/site-packages",)).unwrap();  // append venv path
        path.call_method1("append", ("../bvr_detector_lib_python",)).unwrap();

        for entry in fs::read_dir("../bvr_detector_lib_python").unwrap() {
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

        let img_path = "../bvr_detect/tests/8_people.jpg";

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
