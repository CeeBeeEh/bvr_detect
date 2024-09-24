pub mod detection_processing;
mod utils;
mod extern_c_api;
mod detectors;
pub mod bvr_data;
mod send_channels;

pub mod bvr_detect {
    use crate::{detection_processing, utils};

    use std::{env, sync::mpsc, sync::mpsc::{Receiver, Sender}, sync::LazyLock, thread, time::Instant};
    use log;
    use ort::{inputs, CPUExecutionProvider, CUDAExecutionProvider, Error, ExecutionProvider,
              Session, SessionOutputs, TensorRTExecutionProvider};
    use parking_lot::Mutex;
    use crate::bvr_data::{BvrDetection, BvrImage, DetectorType, DeviceType, ModelConfig};
    use crate::detectors::{detector_onnx, detector_python};
    use crate::send_channels::{DetectionState, SendState};

    pub type Result<T, E = Error> = std::result::Result<T, E>;

    static IS_RUNNING: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));
    static SEND_STATE: LazyLock<Mutex<SendState>> = LazyLock::new(|| Mutex::new({
        let tx = mpsc::channel::<BvrImage>().0;
        let rx2 = mpsc::channel::<Vec<BvrDetection>>().1;
        SendState {
            opt_tx: tx,
            det_rx: rx2,
        }
    }));

    pub async fn is_running() -> Result<bool> {
        let is_running = IS_RUNNING.lock();

        if *is_running { Ok(true) } else { Ok(false) }
    }

    pub async fn init_detector(model_details: ModelConfig, is_test: bool) {
        let mut is_running = IS_RUNNING.lock();
        let mut send_state = SEND_STATE.lock();

        if !*is_running {
            *is_running = true;

            let (det_tx, det_rx) = mpsc::channel::<Vec<BvrDetection>>();
            let (opt_tx, opt_rx) = mpsc::channel::<BvrImage>();

            send_state.det_rx = det_rx;
            send_state.opt_tx = opt_tx;

            let detection_state = DetectionState {
                opt_rx,
                det_tx
            };

            thread::spawn(move || {
                match model_details.detector_type {
                    DetectorType::Onnx => {
                        detector_onnx(is_test, detection_state, model_details).expect("Error in detector_onnx function");
                    }
                    DetectorType::Python => {
                        detector_python(detection_state, model_details).expect("Error in detector_opencv function");
                    }
                };
            });
        } else {
            log::warn!("Detector is already initialized");
        }
    }

    pub async fn detect(bvr_image: BvrImage) -> anyhow::Result<Vec<BvrDetection>> {
        let now = Instant::now();
        let is_running = IS_RUNNING.lock();
        let send_state = SEND_STATE.lock();

        if !*is_running {
            panic!("Detector is not running or hasn't been initialized");
        }

        send_state.opt_tx.send(bvr_image).expect("Error sending opt");

        let detections = send_state.det_rx.recv()?;

        println!("Send time: {:?}", now.elapsed());

        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use crate::bvr_detect::{detect, init_detector};
    use std::path::Path;
    use std::sync::mpsc;
    use std::time::Instant;
    use crate::bvr_data::{BvrDetection, BvrImage, DetectorType, DeviceType, ModelConfig};

    #[tokio::test]
    async fn object_detection() {
        let current_dir = env::current_dir().unwrap();

        /////////////////////
        // Testing variables
        let loop_count: u32 = 5;
        let onnx_path = "../models/yolov9s.onnx".to_string();
        let lib_path= "../onnxruntime_linux_x64_gpu/libonnxruntime.so.1.19.0".to_string();
        let classes_path = "../models/labels_80.txt".to_string();
        let image_path = "../test_images/8_people.jpg";
        /////////////////////

        let model_details = ModelConfig {
            onnx_path,
            lib_path,
            classes_path,
            device_type: DeviceType::CUDA,
            detector_type: DetectorType::Onnx,
            width: 640,
            height: 640,
        };

        let _now = Instant::now();

        let _tx = mpsc::channel::<BvrImage>();
        let _rx = mpsc::channel::<Vec<BvrDetection>>();

        let bvr_image: BvrImage = BvrImage {
            image: image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join(image_path)).unwrap(),
            img_width: 1280,
            img_height: 720,
            conf_thres: 0.7,
            iou_thres: 0.5,
            augment: false,
        };

        init_detector(model_details, true).await;

        let now = Instant::now();
        let mut elapsed = now.elapsed();

        let mut count = 0;

        while count < loop_count {
            let result = detect(bvr_image.clone()).await.unwrap();
            assert_eq!(result.len(), 8);

            let mut detection_thres = String::from("Confidence: ");
            for i in result {
                detection_thres += " | ";
                detection_thres += i.confidence.to_string().as_str();
            }
            println!("\n{}", detection_thres.as_str());
            println!("TIME | Total={:.2?} | {}th detection={:.2?}", now.elapsed(), count, now.elapsed() - elapsed);
            println!("Correctly detected 8 people\n");
            elapsed = now.elapsed();

            count += 1;
        }
    }
}

