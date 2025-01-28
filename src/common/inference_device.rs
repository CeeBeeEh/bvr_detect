#[derive(Debug, Default, Clone, Copy)]
pub enum InferenceDevice {
    #[default] CPU,
    CUDA(usize),
    TensorRT(usize),
    CoreML(usize),
    // TODO: Add/tests more device types (execution providers)
    // ROCm,
}

// Hardcoded device names. Storing the "proper" spelling and the lowercase version.
const CPU: [&str; 2] = ["CPU","cpu"];
const CUDA: [&str; 2] = ["CUDA","cuda"];
const TENSOR_RT: [&str; 2] = ["TensorRT","tensorrt"];
const CORE_ML: [&str; 2] = ["CoreML","coreml"];

impl InferenceDevice {
    pub fn from_str(device: &str, device_id: usize) -> Option<Self> {
        match device.to_lowercase().as_str() {
            "cpu" => Some(InferenceDevice::CPU),
            "cuda" => Some(InferenceDevice::CUDA(device_id)),
            "tensorrt" => Some(InferenceDevice::TensorRT(device_id)),
            "coreml" => Some(InferenceDevice::CoreML(device_id)),
            // Add more cases for other devices as needed
            _ => None,
        }
    }

    pub fn str(&self) -> &'static str {
        match self {
            InferenceDevice::CPU => CPU[0],
            InferenceDevice::CUDA(_) => CUDA[0],
            InferenceDevice::TensorRT(_) => TENSOR_RT[0],
            InferenceDevice::CoreML(_) => CORE_ML[0],
        }
    }

    pub fn str_lowercase(&self) -> &'static str {
        match self {
            InferenceDevice::CPU => CPU[1],
            InferenceDevice::CUDA(_) => CUDA[1],
            InferenceDevice::TensorRT(_) => TENSOR_RT[1],
            InferenceDevice::CoreML(_) => CORE_ML[1],
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            InferenceDevice::CPU => CPU[0].to_string(),
            InferenceDevice::CUDA(_) => CUDA[0].to_string(),
            InferenceDevice::TensorRT(_) => TENSOR_RT[0].to_string(),
            InferenceDevice::CoreML(_) => CORE_ML[0].to_string(),
        }
    }

    pub fn to_lowercase_string(&self) -> String {
        match self {
            InferenceDevice::CPU => CPU[1].to_lowercase(),
            InferenceDevice::CUDA(_) => CUDA[1].to_lowercase(),
            InferenceDevice::TensorRT(_) => TENSOR_RT[1].to_lowercase(),
            InferenceDevice::CoreML(_) => CORE_ML[1].to_lowercase(),
        }
    }

    pub fn all_inference_devices() -> Vec<String> {
        vec![
            InferenceDevice::CPU.to_lowercase_string(),
            InferenceDevice::CUDA(0).to_lowercase_string(),
            InferenceDevice::TensorRT(0).to_lowercase_string(),
            InferenceDevice::CoreML(0).to_lowercase_string(),
        ]
    }

    pub fn is_valid_inference_device(inference_device: String) -> bool {
        match InferenceDevice::from_str(inference_device.to_lowercase().as_str(), 0) {
            Some(_) => { true }
            None => { false }
        }
    }
}