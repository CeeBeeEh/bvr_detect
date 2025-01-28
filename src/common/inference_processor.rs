#[derive(Debug, Default, Clone, Copy)]
pub enum InferenceProcessor {
    #[default] ORT,
    Torch,
    Python,
}

// Hardcoded processor names. Storing the "proper" spelling and the lowercase version.
// The "to_lowercase" is added just to ensure the lowercase version isn't broken by
// human error.
static ORT: [&str; 2] = ["ORT","ort"];
static TORCH: [&str; 2] = ["Torch","torch"];
static PYTHON: [&str; 2] = ["Python","python"];

impl InferenceProcessor {
    pub fn from_str(infer_lang: &str) -> Option<Self> {
        match infer_lang.to_lowercase().as_str() {
            "ort" => Some(InferenceProcessor::ORT),
            "torch_detector" => Some(InferenceProcessor::Torch),
            "python" => Some(InferenceProcessor::Python),
            _ => None,
        }
    }

    pub fn str(&self) -> &'static str {
        match self {
            InferenceProcessor::ORT => ORT[0],
            InferenceProcessor::Torch => TORCH[0],
            InferenceProcessor::Python => PYTHON[0],
        }
    }

    pub fn str_lowercase(&self) -> &'static str {
        match self {
            InferenceProcessor::ORT => ORT[1],
            InferenceProcessor::Torch => TORCH[1],
            InferenceProcessor::Python => PYTHON[1],
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            InferenceProcessor::ORT => ORT[0].to_string(),
            InferenceProcessor::Torch => TORCH[0].to_string(),
            InferenceProcessor::Python => PYTHON[0].to_string(),
        }
    }

    pub fn to_lowercase_string(&self) -> String {
        match self {
            InferenceProcessor::ORT => ORT[1].to_lowercase(),
            InferenceProcessor::Torch => TORCH[1].to_lowercase(),
            InferenceProcessor::Python => PYTHON[1].to_lowercase(),
        }
    }

    pub fn all_inference_processors() -> Vec<String> {
        vec![
            InferenceProcessor::ORT.to_lowercase_string(),
            InferenceProcessor::Torch.to_lowercase_string(),
            InferenceProcessor::Python.to_lowercase_string(),
        ]
    }

    pub fn is_valid_inference_processor(inference_processor: String) -> bool {
        match InferenceProcessor::from_str(inference_processor.to_lowercase().as_str()) {
            Some(_) => { true }
            None => { false }
        }
    }
}