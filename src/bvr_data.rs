use image::{DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone)]
pub struct ModelConfig {
    pub onnx_path: String,
    pub lib_path: String,
    pub classes_path: String,
    pub device_type: DeviceType,
    pub detector_type: DetectorType,
    pub width: u32,
    pub height: u32,
}

impl ModelConfig {
    pub fn new(onnx_path: String, lib_path: String, classes_path: String,
               device_type: DeviceType, detector_type: DetectorType,
               width: u32, height: u32) -> Self {
        Self {
            onnx_path,
            lib_path,
            classes_path,
            device_type,
            detector_type,
            width,
            height,
        }
    }

    pub fn set_device_type(&mut self, device_type: DeviceType) {
        self.device_type = device_type;
    }

    pub fn to_string(&self) -> String {
        format!("Onnx Path: {}\nClasses Path: {}\nDevice Type (execution provider): {:?}\nModel input resolution: {}x{}",
                 self.onnx_path, self.classes_path, self.device_type, self.width, self.height)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum DetectorType {
    #[default] Onnx,
    Python,
}

impl DetectorType {
    pub fn from_str(infer_lang: &str) -> Option<Self> {
        match infer_lang.to_lowercase().as_str() {
            "native" => Some(DetectorType::Onnx),
            "python" => Some(DetectorType::Python),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DetectorType::Onnx => "Native",
            DetectorType::Python => "Python",
        }
    }

    pub fn as_str_lowercase(&self) -> &'static str {
        match self {
            DetectorType::Onnx => "native",
            DetectorType::Python => "python",
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum DeviceType {
    #[default] CPU,
    CUDA,
    TensorRT,
    // TODO: Add/test more device types (execution providers)
    //ROCm,
}

impl DeviceType {
    pub fn from_str(device: &str) -> Option<Self> {
        match device.to_lowercase().as_str() {
            "cpu" => Some(DeviceType::CPU),
            "cuda" => Some(DeviceType::CUDA),
            "tensorrt" => Some(DeviceType::TensorRT),
            // Add more cases for other devices as needed
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceType::CPU => "CPU",
            DeviceType::CUDA => "CUDA",
            DeviceType::TensorRT => "TensorRT",
        }
    }

    pub fn as_str_lowercase(&self) -> &'static str {
        match self {
            DeviceType::CPU => "cpu",
            DeviceType::CUDA => "cuda",
            DeviceType::TensorRT => "tensorrt",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BvrImage {
    pub image: DynamicImage,
    pub img_width: i32,
    pub img_height: i32,
    pub conf_thres: f64,
    pub iou_thres: f64,
    pub augment: bool,
}

impl BvrImage {
    pub fn new(image: DynamicImage, conf_thres: f64, iou_thres: f64, augment: bool) -> Self {
        let (img_width, img_height) = image.dimensions();
        Self {
            image,
            img_width: img_width as i32,
            img_height: img_height as i32,
            conf_thres,
            iou_thres,
            augment,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BvrBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
    pub width: i32,
    pub height: i32,
}

impl BvrBox {
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            width: x2 - x1,
            height: y2 - y1,
        }
    }

    pub fn area(&self) -> i32 {
        self.width * self.height
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x1 && x <= self.x2 && y >= self.y1 && y <= self.y2
    }

    pub fn scale(&mut self, factor: f32) {
        self.width = (self.width as f32 * factor) as i32;
        self.height = (self.height as f32 * factor) as i32;
        self.x2 = self.x1 + self.width;
        self.y2 = self.y1 + self.height;
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct BvrDetection {
    pub id: u32,
    pub class_id: u32,
    pub track_id: u32,
    pub bbox: BvrBox,
    pub label: String,
    pub last_inference_time: u128,
    pub inference_hits: u32,
    pub frame_hits: u32,
    pub confidence: f32,
    pub num_matches: u32,
    pub camera_name: String,
    pub camera_id: u32,
}

impl BvrDetection {
    pub fn new(class_id: u32, bbox: BvrBox, label: String, confidence: f32) -> Self {
        Self {
            id: 0,
            class_id,
            track_id: 0,
            bbox,
            label,
            last_inference_time: 0,
            inference_hits: 0,
            frame_hits: 0,
            confidence,
            num_matches: 0,
            camera_name: String::new(),
            camera_id: 0,
        }
    }

    pub fn inference_hit(&mut self) {
        self.inference_hits += 1;
    }

    pub fn frame_hit(&mut self) {
        self.inference_hits += 1;
    }

    pub fn print_detection(&self) {
        println!(
            "Detection: ID: {}, Class: {}, BBox: {:?}, Confidence: {:.2}",
            self.id, self.class_id, self.bbox, self.confidence
        );
    }
}