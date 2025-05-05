use crate::common::{ModelVersion, YoloPreds};
use crate::data::{DynConf, MinOptMax};
use crate::detection_runners::OrtEngine;

#[derive(Debug)]
pub struct BvrOrtYOLO {
    pub(crate) engine: OrtEngine,
    pub(crate) nc: usize,
    pub(crate) height: MinOptMax,
    pub(crate) width: MinOptMax,
    pub(crate) batch: MinOptMax,
    pub(crate) confs: DynConf,
    pub(crate) iou: f32,
    pub(crate) names: Vec<String>,
    pub(crate) layout: YoloPreds,
    pub(crate) version: Option<ModelVersion>,
}
