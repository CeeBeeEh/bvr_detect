use std::time::Instant;
use image::GenericImageView;
use image::imageops::FilterType;
use ndarray::{Array, Ix4, IxDyn};
use crate::bvr_data::{BvrBox, BvrDetection, BvrImage};

fn intersection(box1: &BvrBox, box2: &BvrBox) -> i32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BvrBox, box2: &BvrBox) -> i32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}

pub fn process_image(bvr_image: BvrImage, width: u32, height: u32) -> (u32, u32, Array<f32, Ix4>) {
    let (img_width, img_height) = (bvr_image.image.width(), bvr_image.image.height());

    let mut resizer = fast_image_resize::Resizer::new();
    let options = fast_image_resize::ResizeOptions {
        algorithm: fast_image_resize::ResizeAlg::Convolution(
            fast_image_resize::FilterType::Bilinear,
        ),
        ..Default::default()
    };

    let mut new_image = image::DynamicImage::new(width, height, bvr_image.image.color());
    if let Err(err) = resizer.resize(&bvr_image.image, &mut new_image, &options) {
        tracing::warn!(?err, "Failed to use `fast_image_resize`. Falling back.");
        new_image =
            image::imageops::resize(&bvr_image.image, width, height, FilterType::Nearest).into();
    }
    let mut input: Array<f32, Ix4> = Array::zeros((1, 3, width as usize, height as usize));

    // TODO: This can be faster
    for pixel in new_image.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    (img_width, img_height, input)
}

pub(crate) fn process_predictions(output: &Array<f32, IxDyn>, classes_list: &Vec<String>,
                                  width_f32: f32, height_f32: f32,
                                  img_width: f32, img_height: f32,
                                  output_shape: &[usize], detection_time: u128) -> Vec<BvrDetection> {
    let mut boxes = Vec::new();

    let now = Instant::now();

    let mut rowc: Vec<u128> = Vec::new();
    let mut boxc: Vec<u128> = Vec::new();
    let mut boxp: Vec<u128> = Vec::new();

    let reshaped_output = output
        .to_shape((output_shape[0], output_shape[1]))
        .expect("Failed to reshape the output");
    for detection in reshaped_output.outer_iter() {
        let mut _detect_elapsed = now.elapsed();

        let prob = detection[4];
        let class_id = detection[5] as i32;

        rowc.push((now.elapsed() - _detect_elapsed).as_nanos());
        _detect_elapsed = now.elapsed();

        if prob < 0.5 {
            continue;
        }

        let label = classes_list[class_id as usize].as_str();
        let xc = detection[0] / width_f32 * img_width;
        let yc = detection[1] / height_f32 * img_height;
        let w = detection[2] / width_f32 * img_width;
        let h = detection[3] / height_f32 * img_height;
        boxc.push((now.elapsed() - _detect_elapsed).as_nanos());
        _detect_elapsed = now.elapsed();

        boxes.push((
            BvrBox {
                x1: (xc - w / 2.0).round() as i32,
                y1: (yc - h / 2.0).round() as i32,
                x2: (xc + w / 2.0).round() as i32,
                y2: (yc + h / 2.0).round() as i32,
                width: w.round() as i32,
                height: h.round() as i32,
            },
            label,
            prob,
        ));
        boxp.push((now.elapsed() - _detect_elapsed).as_nanos());
        _detect_elapsed = now.elapsed();
    }

    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));

    let mut detections: Vec<BvrDetection> = Vec::new();

    while !boxes.is_empty() {
        let i = boxes.remove(0);
        let mut det: BvrDetection = Default::default();
        det.bbox = i.0;
        //det.label = classes_list[i.1];
        det.confidence = i.2;
        det.last_inference_time = detection_time;
        detections.push(det);

        boxes = boxes
            .iter()
            .filter(|box1| (intersection(&boxes[0].0, &box1.0) as f32 / union(&boxes[0].0, &box1.0) as f32) < 0.7)
            .copied()
            .collect();
    }

    detections
}