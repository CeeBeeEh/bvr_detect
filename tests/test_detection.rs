extern crate bvr_detect;

use std::future::Future;
use std::path::Path;
use std::time::Instant;
use ab_glyph::{FontRef, PxScale};
use image::GenericImageView;
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use bvr_detect::common::{BvrDetection, BvrImage, InferenceDevice, InferenceProcessor, ModelConfig, ModelVersion};

mod colours;

#[cfg(test)]
#[tokio::test]
async fn detection() {
    /////////////////////
    // Testing variables
    let loop_count: u32 = 10;
    let onnx_path = "../models/yolov11/yolo11n.onnx".to_string();
    let lib_path= "../onnxruntime/linux_x64_gpu/libonnxruntime.so.1.20.1".to_string();
    let classes_path = "../models/labels_80.txt".to_string();
    let image_path = "tests/8_people.jpg";
    //let image_path = "../test_images/signal-2024-09-26-150939_003.jpg";
    let yolo_ver = "yolov11".to_string();
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
        width: 960,
        height: 960,
        rect: false,
        split_wide_input: true,
    };

    let _now = Instant::now();

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

    let mut yolo = match bvr_detect::init_detector(&model_details, true) {
        Ok(yolo) => yolo,
        _ => panic!("Failed to initialize YOLO model")
    };

    let now = Instant::now();
    let mut elapsed = now.elapsed();

    let mut count = 0;

    while count < loop_count {
        let result = bvr_detect::run_detection(&mut yolo, bvr_image.clone(), &model_details).unwrap();
        //assert_eq!(result.len(), 8);

        let mut detection_thres = String::from("Confidence: ");
        let detection_count = result.len();

        if detection_count > 0 && count == 0 {
            let mut img = bvr_image.image.clone();

            let font = FontRef::try_from_slice(include_bytes!("DejaVuSansMono.ttf")).unwrap();
            for i in &result {
                let (x, y, w, h) = i.bbox.as_xy_wh_i32();
                let rect = Rect::at(x, y).of_size(w as u32, h as u32);
                let draw_color = colours::get_class_colour(i.class_id as usize);
                draw_hollow_rect_mut(&mut img, rect, draw_color);

                let height = 20.;
                let scale = PxScale {
                    x: height * 2.0,
                    y: height,
                };

                draw_text_mut(&mut img, draw_color, x, y, scale, &font, i.get_label().as_str())
            }

            img.save("tests/test_output.jpg").unwrap();
        }

        for i in result {
            detection_thres += " | ";
            detection_thres += i.confidence.to_string().as_str();
        }
        println!("\n{}", detection_thres.as_str());
        println!("TIME | Total={:.2?} | {}th detection_runners={:.2?}", now.elapsed(), count, now.elapsed() - elapsed);
        println!("Detected {} objects\n", detection_count);
        elapsed = now.elapsed();

        count += 1;
    }
}

