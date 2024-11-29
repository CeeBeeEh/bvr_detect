use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct LabelThreshold {
    pub id: u16,
    pub label: String,
    pub threshold: f32,
}

impl LabelThreshold {
    pub fn new(id: u16, label: String, threshold: f32) -> Self {
        Self {
            id,
            label,
            threshold,
        }
    }
    
    pub fn check_conf(self, conf: f32) -> bool {
        self.threshold > conf
    }
    
    pub fn check_label_conf(self, other: &LabelThreshold) -> bool {
        if self.label != other.label || self.id != other.id {
            return false;
        }
        self.threshold >= other.threshold
    }
}