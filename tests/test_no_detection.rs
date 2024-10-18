use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;
use image::GenericImageView;
use BvrDetect::bvr_detect::{detect, init_detector};
use BvrDetect::data::{BvrDetection, BvrImage, DeviceType, ModelConfig, ProcessingType, YoloVersion};

#[tokio::test]
fn no_detections() {
    /////////////////////
    // Testing variables
    let loop_count: u32 = 3;
    let onnx_path = "../models/yolov11/yolo11n.onnx".to_string();
    let lib_path= "../ort_detector/linux_x64_gpu/libonnxruntime.so.1.19.0".to_string();
    let classes_path = "../models/labels_80.txt".to_string();
    let image_path = "blank.jpg";
    let yolo_ver = "v11".to_string();
    /////////////////////

    let yolo_version = YoloVersion::from(yolo_ver);

    let model_details = ModelConfig {
        weights_path: onnx_path,
        ort_lib_path: lib_path,
        labels_path: classes_path,
        device_type: DeviceType::CUDA(0),
        processing_type: ProcessingType::ORT,
        yolo_version,
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
        img_width: img_width as i32,
        img_height: img_height as i32,
        threshold: 0.5,
        augment: false,
    };

    let _ = init_detector(model_details, true);

    let now = Instant::now();
    let mut elapsed = now.elapsed();

    let mut count = 0;

    while count < loop_count {
        let result = detect(bvr_image.clone()).unwrap();
        assert_eq!(result.len(), 0);

        println!("TIME | Total={:.2?} | {}th detection_runners={:.2?}", now.elapsed(), count, now.elapsed() - elapsed);
        println!("Detected 0 objects\n");
        elapsed = now.elapsed();

        count += 1;
    }
}

