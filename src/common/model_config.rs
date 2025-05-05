use crate::common::inference_device::InferenceDevice;
use crate::common::inference_processor::InferenceProcessor;
use crate::common::model_version::ModelVersion;

#[derive(Default, Debug, Clone)]
pub struct ModelConfig {
    pub weights_path: String,
    pub ort_lib_path: String,
    // TODO: pub torch_lib_path: String,
    // TODO: pub python_lib_path: String,
    pub labels_path: String,
    pub inference_device: InferenceDevice,
    pub inference_processor: InferenceProcessor,
    pub model_version: ModelVersion,
    pub conf_threshold: f32,
    pub rect: bool,
    pub width: u32,
    pub height: u32,
    pub split_wide_input: bool,
}

impl ModelConfig {
    pub fn new(weights_path: String, ort_lib_path: String, labels_path: String,
               inference_device: InferenceDevice, inference_processor: InferenceProcessor,
               model_version: ModelVersion, conf_threshold: f32, rect: bool,
               width: u32, height: u32, split_wide_input: bool) -> Self {
        Self {
            weights_path,
            ort_lib_path,
            labels_path,
            inference_device,
            inference_processor,
            model_version,
            conf_threshold,
            width,
            height,
            rect,
            split_wide_input,
        }
    }
    
    pub fn empty() -> Self {
        Self {
            weights_path: "".to_string(),
            ort_lib_path: "".to_string(),
            labels_path: "".to_string(),
            inference_device: Default::default(),
            inference_processor: Default::default(),
            model_version: Default::default(),
            conf_threshold: 0.4,
            rect: false,
            width: 960,
            height: 960,
            split_wide_input: false,
        }
    }

    pub fn set_device_type(&mut self, device_type: InferenceDevice) {
        self.inference_device = device_type;
    }

    pub fn to_string(&self) -> String {
        format!("Weights File Path: {}\n\
        Labels Path: {}\n\
        OnnxRuntime Lib Path: {}\n\
        Inference Device: {:?}\n\
        Inference Processor (execution provider): {:?}\n\
        Model Version: {:?}\n\
        Model Input Size: {}x{}\n\
        Rect input: {}\n\
        Detection Threshold: {}",
                self.weights_path, self.labels_path, self.ort_lib_path,
                self.inference_device, self.inference_processor, self.model_version,
                self.width, self.height, self.rect, self.conf_threshold)
    }
    
    pub fn get_threshold(&self) -> f32 {
        &self.conf_threshold * 1.0      // somehow I can only get the borrower to be quiet by multiplying by 1
    }
}