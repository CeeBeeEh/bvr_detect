use std::env;
use std::time::Instant;
use image::{DynamicImage, GenericImageView};
use ort::{inputs, CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider, Session, SessionOutputs, TensorRTExecutionProvider};
//use pyo3::{types, PyResult, Python};
//use pyo3::prelude::*;
//use pyo3::types::{PyBytes, PyModule};
use crate::{detection_processing, utils};
use crate::bvr_data::{BvrDetection, DeviceType, ModelConfig};
use crate::send_channels::DetectionState;

pub fn detector_onnx(is_test: bool, detection_state: DetectionState, model_details: ModelConfig) -> anyhow::Result<()> {
    //OnceCell::new()

    // Dynamically load the library from given path
    ort::init_from(model_details.lib_path).commit()?;

    let session_builder = Session::builder()?;

    let execution_provider = match model_details.device_type {
        DeviceType::CPU => { CPUExecutionProvider::default().build() },
        DeviceType::CUDA => {
            let cuda = CUDAExecutionProvider::default();
            match cuda.register(&session_builder) {
                Ok(_) => log::info!("CUDA device successfully registered"),
                Err(e) => {
                    log::warn!("Failed to register CUDA device: {}", e);
                    std::process::exit(1);
                }
            }
            cuda.build()
        },
        DeviceType::TensorRT => {
            let tensor_rt = TensorRTExecutionProvider::default();
            match tensor_rt.register(&session_builder) {
                Ok(_) => log::info!("TensorRT device successfully registered"),
                Err(e) => {
                    log::warn!("Failed to register TensorRT device: {}", e);
                    std::process::exit(1);
                }
            }
            tensor_rt.build()
        },
    };

    let classes_list = utils::file_to_vec(model_details.classes_path.to_string())?;
    let session = session_builder.commit_from_file(model_details.onnx_path)?;

    let input_info = &session.inputs;
    let output_info = &session.outputs;

    log::info!("Initializing ORT session with ({}) execution provider", model_details.device_type.as_str());
    ort::init()
        .with_execution_providers([execution_provider])
        .commit()?;

    loop {
        // MESSAGE LOOP STARTS HERE
        let bvr_image = match detection_state.opt_rx.recv() {
            Ok(msg) => msg,
            Err(err) => {
                log::error!("bvr_detect: Failed to receive image: {}", err);
                continue;
            }
        };
        let detect_time = Instant::now();

        let (img_width, img_height, input) = detection_processing::process_image(bvr_image, model_details.width, model_details.height);
        let mut _detect_elapsed = detect_time.elapsed();

        _detect_elapsed = utils::trace(is_test, "TIME", "Preprocessing input", detect_time, _detect_elapsed);

        // Run ONNX inference
        let outputs: SessionOutputs = session.run(inputs![&input_info[0].name => input.view()]?)?;
        let detection_time = (detect_time.elapsed() - _detect_elapsed).as_micros();

        _detect_elapsed = utils::trace(is_test, "TIME", "Detection run", detect_time, _detect_elapsed);

        let output = outputs[output_info[0].name.as_str()].try_extract_tensor::<f32>()?.t().into_owned();

        let output_shape = output.shape();

        _detect_elapsed = utils::trace(is_test, "TIME", "Outputs", detect_time, _detect_elapsed);

        let detections = detection_processing::process_predictions(&output, &classes_list,
                                                                   model_details.width as f32, model_details.height as f32,
                                                                   img_width as f32, img_height as f32,
                                                                   output_shape, detection_time);

        _detect_elapsed = utils::trace(is_test, "TIME", "Postprocessing", detect_time, _detect_elapsed);

        _detect_elapsed = utils::trace(is_test, "TIME", "NMS", detect_time, _detect_elapsed);

        detection_state.det_tx.send(detections)?;
    }
}

pub fn detector_python(detection_state: DetectionState, model_details: ModelConfig) -> anyhow::Result<()> {
/*    let python_file = std::fs::read_to_string("/mnt/4TB/Development/Bvr-Project/bvr_detector_lib_python/detector_2.py")?;

    Python::with_gil(|py| -> PyResult<()> {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", ("/mnt/4TB/Development/Bvr-Project/bvr_detector_lib_python/venv/lib/python3.12/site-packages/",))?;  // append my venv path
        path.call_method1("append", ("/mnt/4TB/Development/Bvr-Project/bvr_detector_lib_python/",))?;  // append my venv path

        let p_module = PyModule::from_code_bound(py, &python_file, "detector.py", "detector")?;

        let device = "cpu"; // Or "cuda" depending on your setup
        let model_size = (model_details.width, model_details.height);
        let conf_thresh = 0.5;
        let iou_thresh = 0.4;

        let init_detector: bool = p_module.getattr("init_detector")?.call1((device, model_details.onnx_path, model_size, conf_thresh, iou_thresh))?.extract()?;

        if init_detector == false {
            println!("No Beuno Senior!")
        }

        // Call run_detector method
        let run_detector = p_module.getattr("run_detector").unwrap();

        loop {
            // MESSAGE LOOP STARTS HERE
            let bvr_image = match detection_state.opt_rx.recv() {
                Ok(msg) => msg,
                Err(err) => {
                    log::error!("bvr_detect: Failed to receive image: {}", err);
                    continue;
                }
            };

            // Convert image data to PyBytes
            let py_image_data = PyBytes::new_bound(py, bvr_image.image.as_bytes());

            // Call run_detector with the image data
            let result = run_detector.call1((py_image_data,)).unwrap();

            let json_str: &str = result.extract().unwrap();

            let detections: Vec<BvrDetection> = serde_json::from_str(json_str).unwrap();


            println!("Result: {}", result);

            detection_state.det_tx.send(detections).unwrap();
        }
    }).expect("TODO: panic message");
*/
    Ok(())
}

