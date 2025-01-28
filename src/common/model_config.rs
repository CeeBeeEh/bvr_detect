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
    pub width: u32,
    pub height: u32,
    pub split_wide_input: bool,
}

impl ModelConfig {
    pub fn new(weights_path: String, ort_lib_path: String, labels_path: String,
               inference_device: InferenceDevice, inference_processor: InferenceProcessor,
               model_version: ModelVersion, conf_threshold: f32,
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
            split_wide_input,
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
        Model Input Resolution: {}x{}\n\
        Detection Threshold: {}",
                self.weights_path, self.labels_path, self.ort_lib_path,
                self.inference_device, self.inference_processor, self.model_version,
                self.width, self.height, self.conf_threshold)
    }
}