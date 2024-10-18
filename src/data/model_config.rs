use crate::data::{DeviceType, ProcessingType, YoloVersion};

#[derive(Default, Debug, Clone)]
pub struct ModelConfig {
    pub weights_path: String,
    pub ort_lib_path: String,
    // TODO: pub torch_lib_path: String,
    // TODO: pub python_lib_path: String,
    pub labels_path: String,
    pub device_type: DeviceType,
    pub processing_type: ProcessingType, // I'm not happy with this name
    pub yolo_version: YoloVersion,
    pub conf_threshold: f32,
    pub width: u32,
    pub height: u32,
    pub split_wide_input: bool,
}

impl ModelConfig {
    pub fn new(weights_path: String, ort_lib_path: String, labels_path: String,
               device_type: DeviceType, processing_type: ProcessingType,
               yolo_version: YoloVersion, conf_threshold: f32,
               width: u32, height: u32, split_wide_input: bool) -> Self {
        Self {
            weights_path,
            ort_lib_path,
            labels_path,
            device_type,
            processing_type,
            yolo_version,
            conf_threshold,
            width,
            height,
            split_wide_input,
        }
    }

    pub fn set_device_type(&mut self, device_type: DeviceType) {
        self.device_type = device_type;
    }

    pub fn to_string(&self) -> String {
        format!("Weights File Path: {}\n\
        Labels Path: {}\n\
        OnnxRuntime Lib Path: {}\n\
        Device Type (execution provider): {:?}\n\
        Model Input Resolution: {}x{}\n\
        Detection Threshold: {}",
                self.weights_path, self.labels_path, self.ort_lib_path,
                self.device_type, self.width, self.height, self.conf_threshold)
    }
}