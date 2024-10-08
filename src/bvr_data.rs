use image::{DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone)]
pub struct ModelConfig {
    pub onnx_path: String,
    pub ort_lib_path: String,
    pub classes_path: String,
    pub device_type: DeviceType,
    pub detector_type: ProcessingType,
    pub threshold: f32,
    pub width: u32,
    pub height: u32,
}

impl ModelConfig {
    pub fn new(onnx_path: String, ort_lib_path: String, classes_path: String,
               device_type: DeviceType, detector_type: ProcessingType,
               threshold: f32, width: u32, height: u32) -> Self {
        Self {
            onnx_path,
            ort_lib_path,
            classes_path,
            device_type,
            detector_type,
            threshold,
            width,
            height,
        }
    }

    pub fn set_device_type(&mut self, device_type: DeviceType) {
        self.device_type = device_type;
    }

    pub fn to_string(&self) -> String {
        format!("Onnx Path: {}\n\
        Classes Path: {}\n\
        OnnxRuntime Lib Path: {}\n\
        Device Type (execution provider): {:?}\n\
        Model Input Resolution: {}x{}\n\
        Detection Threshold: {}",
        self.onnx_path, self.classes_path, self.ort_lib_path,
        self.device_type, self.width, self.height, self.threshold)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum ProcessingType {
    #[default] Native,
    Python,
}

impl ProcessingType {
    pub fn from_str(infer_lang: &str) -> Option<Self> {
        match infer_lang.to_lowercase().as_str() {
            "native" => Some(ProcessingType::Native),
            "python" => Some(ProcessingType::Python),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ProcessingType::Native => "Native",
            ProcessingType::Python => "Python",
        }
    }

    pub fn as_str_lowercase(&self) -> &'static str {
        match self {
            ProcessingType::Native => "native",
            ProcessingType::Python => "python",
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
    pub threshold: f32,
    pub augment: bool,
}

impl BvrImage {
    pub fn new(image: DynamicImage, threshold: f32, augment: bool) -> Self {
        let (img_width, img_height) = image.dimensions();
        Self {
            image,
            img_width: img_width as i32,
            img_height: img_height as i32,
            threshold,
            augment,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BvrBoxF32 {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BvrBoxF32 {
    pub fn to_bvr_box(&self) -> BvrBox {
        BvrBox {
            x1: self.x1.round() as i32,
            y1: self.y1.round() as i32,
            x2: self.x2.round() as i32,
            y2: self.y2.round() as i32,
            w: self.x2.round() as i32 - self.x1.round() as i32,
            h: self.y2.round() as i32 - self.y1.round() as i32,
        }
    }
}
#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BvrBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
    pub w: i32,
    pub h: i32,
}

impl BvrBox {
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            w: x2 - x1,
            h: y2 - y1,
        }
    }

    pub fn area(&self) -> i32 {
        self.w * self.h
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x1 && x <= self.x2 && y >= self.y1 && y <= self.y2
    }

    pub fn scale(&mut self, factor: f32) {
        self.w = (self.w as f32 * factor) as i32;
        self.h = (self.h as f32 * factor) as i32;
        self.x2 = self.x1 + self.w;
        self.y2 = self.y1 + self.h;
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