use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;
use image::GenericImageView;
use bvr_detect::common::{BvrDetection, BvrImage, InferenceDevice, InferenceProcessor, ModelConfig, ModelVersion};

#[tokio::test]
async fn no_detections() {
    /////////////////////
    // Testing variables
    let loop_count: u32 = 3;
    let onnx_path = "../models/yolov11/yolo11n.onnx".to_string();
    let lib_path= "../ort_detector/linux_x64_gpu/libonnxruntime.so.1.19.0".to_string();
    let classes_path = "../models/labels_80.txt".to_string();
    let image_path = "blank.jpg";
    let yolo_ver = "v11".to_string();
    /////////////////////

    let model_version = ModelVersion::from(yolo_ver);

    let model_details = ModelConfig {
        weights_path: onnx_path,
        ort_lib_path: lib_path,
        labels_path: classes_path,
        inference_device: InferenceDevice::CUDA(0),
        inference_processor: InferenceProcessor::ORT,
        model_version,
        conf_threshold: 0.3,
        width: 640,
        height: 640,
        split_wide_input: true,
    };

    let _now = Instant::now();

    let _tx = mpsc::channel::<BvrImage>();
    let _rx = mpsc::channel::<Vec<BvrDetection>>();

    let image = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join(image_path)).unwrap();
    let (img_width, img_height) = image.dimensions();

    let bvr_image: BvrImage = BvrImage {
        image,
        img_width,
        img_height,
        threshold: 0.5,
        augment: false,
        wanted_labels: None,
    };

    let _ = bvr_detect::init_detector(model_details, true);

    let now = Instant::now();
    let mut elapsed = now.elapsed();

    let mut count = 0;

    while count < loop_count {
        let result = bvr_detect::run_detection(bvr_image.clone()).await.unwrap();
        assert_eq!(result.len(), 0);

        println!("TIME | Total={:.2?} | {}th detection_runners={:.2?}", now.elapsed(), count, now.elapsed() - elapsed);
        println!("Detected 0 objects\n");
        elapsed = now.elapsed();

        count += 1;
    }
}

