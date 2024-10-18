#[derive(Debug, Default, Clone, Copy)]
pub enum DeviceType {
    #[default] CPU,
    CUDA(usize),
    TensorRT(usize),
    CoreML(usize),
    // TODO: Add/tests more device types (execution providers)
    //ROCm,
}

impl DeviceType {
    pub fn from_str(device: &str, device_id: usize) -> Option<Self> {
        match device.to_lowercase().as_str() {
            "cpu" => Some(DeviceType::CPU),
            "cuda" => Some(DeviceType::CUDA(device_id)),
            "tensorrt" => Some(DeviceType::TensorRT(device_id)),
            "coreml" => Some(DeviceType::CoreML(device_id)),
            // Add more cases for other devices as needed
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceType::CPU => "CPU",
            DeviceType::CUDA(_) => "CUDA",
            DeviceType::TensorRT(_) => "TensorRT",
            DeviceType::CoreML(_) => "CoreML",
        }
    }

    pub fn as_str_lowercase(&self) -> &'static str {
        match self {
            DeviceType::CPU => "cpu",
            DeviceType::CUDA(_) => "cuda",
            DeviceType::TensorRT(_) => "tensorrt",
            DeviceType::CoreML(_) => "coreml",
        }
    }
}