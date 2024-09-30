pub mod detection_processing;
mod utils;
mod extern_c_api;
mod detectors;
pub mod bvr_data;
mod send_channels;

pub mod bvr_detect {
    use std::{sync::LazyLock, thread, time::Instant};
    //    use std::cell::OnceCell;
    use log;
    use ort::Error;
    use parking_lot::Mutex;
    use crate::bvr_data::{BvrDetection, BvrImage, ProcessingType, ModelConfig};
    use crate::detectors::detector_onnx;
    use crate::send_channels::{DetectionState, SendState};

    pub type Result<T, E = Error> = std::result::Result<T, E>;

    static IS_RUNNING: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));
    static SEND_STATE: LazyLock<Mutex<SendState>> = LazyLock::new(|| Mutex::new({
        let tx = crossbeam_channel::unbounded::<Box<BvrImage>>().0;
        let rx2 = crossbeam_channel::unbounded::<Box<Vec<BvrDetection>>>().1;
        SendState {
            opt_tx: tx,
            det_rx: rx2,
        }
    }));

//    static SEND_STATE: OnceCell<Mutex<SendState>> = OnceCell::new();

    pub async fn is_running() -> Result<bool> {
        let is_running = IS_RUNNING.lock();

        if *is_running { Ok(true) } else { Ok(false) }
    }

    pub async fn init_detector(model_details: ModelConfig, is_test: bool) {
        let mut is_running = IS_RUNNING.lock();
        let mut send_state = SEND_STATE.lock();

        if !*is_running {
            *is_running = true;

            let (det_tx, det_rx) = crossbeam_channel::unbounded::<Box<Vec<BvrDetection>>>();
            let (opt_tx, opt_rx) = crossbeam_channel::unbounded::<Box<BvrImage>>();

            send_state.det_rx = det_rx;
            send_state.opt_tx = opt_tx;

            let detection_state = DetectionState {
                opt_rx,
                det_tx
            };

            thread::spawn(move || {
                match model_details.detector_type {
                    ProcessingType::Native => {
                        detector_onnx(is_test, detection_state, model_details).expect("Error in detector_onnx function");
                    }
                    ProcessingType::Python => {
                        // DO NOTHING FOR NOW
                        //detector_python(detection_state, model_details).expect("Error in detector_opencv function");
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

        send_state.opt_tx.send(Box::from(bvr_image))?;

        // TODO: We're losing about 5ms on this receive statement for some reason
        let detections = *(send_state.det_rx.recv()?);

        println!("Processing time: {:?}", now.elapsed());

        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use crate::bvr_detect::{detect, init_detector};
    use std::path::Path;
    use std::sync::mpsc;
    use std::time::Instant;
    use image::Rgba;
    use imageproc::drawing::draw_hollow_rect_mut;
    use imageproc::rect::Rect;
    use crate::bvr_data::{BvrDetection, BvrImage, ProcessingType, DeviceType, ModelConfig};

    #[tokio::test]
    async fn object_detection() {
        /////////////////////
        // Testing variables
        let loop_count: u32 = 10;
        let onnx_path = "../models/yolov9s.onnx".to_string();
        let lib_path= "../onnxruntime/linux_x64_gpu/libonnxruntime.so.1.19.0".to_string();
        let classes_path = "../models/labels_80.txt".to_string();
        let image_path = "../test_images/8_people.jpg";
        /////////////////////

        let model_details = ModelConfig {
            onnx_path,
            ort_lib_path: lib_path,
            classes_path,
            device_type: DeviceType::CUDA,
            detector_type: ProcessingType::Native,
            threshold: 0.4,
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
            threshold: 0.5,
            augment: false,
        };

        init_detector(model_details, true).await;

        let now = Instant::now();
        let mut elapsed = now.elapsed();

        let mut count = 0;

        while count < loop_count {
            let result = detect(bvr_image.clone()).await.unwrap();
            //assert_eq!(result.len(), 8);

            let mut detection_thres = String::from("Confidence: ");
            let detection_count = result.len();

            if detection_count > 0 && count == 0 {
                let mut img = bvr_image.image.clone();

                for i in &result {
                    let rect = Rect::at(i.bbox.x1, i.bbox.y1).of_size(i.bbox.w as u32, i.bbox.h as u32);
                    let draw_color = Rgba([255, 0, 0, 255]);
                    draw_hollow_rect_mut(&mut img, rect, draw_color);
                }

                img.save("test_output.jpg").unwrap();
            }

            for i in result {
                detection_thres += " | ";
                detection_thres += i.confidence.to_string().as_str();
            }
            println!("\n{}", detection_thres.as_str());
            println!("TIME | Total={:.2?} | {}th detection={:.2?}", now.elapsed(), count, now.elapsed() - elapsed);
            println!("Detected {} objects\n", detection_count);
            elapsed = now.elapsed();

            count += 1;
        }
    }
}

