mod utils;
mod detectors;
pub mod data;
mod detection_runners;

pub mod bvr_detect {
    use std::{sync::LazyLock, thread, time::Instant};
    //    use std::cell::OnceCell;
    use log;
    use ort::Error;
    use parking_lot::Mutex;
    use crate::data::{BvrDetection, BvrImage, ModelConfig, ProcessingType};
    use crate::detectors::{detector_onnx};
    use crate::data::send_channels::{DetectionState, SendState};

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
                match model_details.processing_type {
                    ProcessingType::ORT => {
                        detector_onnx(is_test, detection_state, model_details).expect("Error in detector_onnx function");
                    }
                    ProcessingType::Torch => {
                        // DO NOTHING FOR NOW
                    }
                    ProcessingType::Python => {
                        //detector_python(is_test, detection_state, model_details).expect("Error in detector_python function");
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

        // TODO: We're losing about 5ms on this receive statement for some reason, aside from the inference/processing time
        let detections = *(send_state.det_rx.recv()?);

        println!("Processing time: {:?}", now.elapsed());

        Ok(detections)
    }
}


