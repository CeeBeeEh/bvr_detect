#[derive(Debug, Default, Clone, Copy)]
pub enum ProcessingType {
    #[default] ORT,
    Torch,
    Python,
}

impl ProcessingType {
    pub fn from_str(infer_lang: &str) -> Option<Self> {
        match infer_lang.to_lowercase().as_str() {
            "ort" => Some(ProcessingType::ORT),
            "torch_detector" => Some(ProcessingType::Torch),
            "python" => Some(ProcessingType::Python),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ProcessingType::ORT => "ORT",
            ProcessingType::Torch => "Torch",
            ProcessingType::Python => "Python",
        }
    }

    pub fn as_str_lowercase(&self) -> &'static str {
        match self {
            ProcessingType::ORT => "ort",
            ProcessingType::Torch => "torch_detector",
            ProcessingType::Python => "python",
        }
    }
}