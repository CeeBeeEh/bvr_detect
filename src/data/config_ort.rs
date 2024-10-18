//! File/code adapted from https://github.com/jamjamjon/usls
//!
//! Options for building models.

use anyhow::Result;

use crate::data::{DeviceType, MinOptMax};
use crate::data::yolo_types::{YoloPreds, YoloVersion};
use crate::detection_runners::Iiix;

#[derive(Debug, Clone)]
pub struct ConfigOrt {
    pub onnx_path: String,
    pub ort_lib_path: String,
    pub device: DeviceType,
    pub batch_size: usize,
    pub iiixs: Vec<Iiix>,
    pub profile: bool,
    pub num_dry_run: usize,

    // trt related
    pub trt_engine_cache_enable: bool,
    pub trt_int8_enable: bool,
    pub trt_fp16_enable: bool,

    // options for Vision and Language models
    pub nc: Option<usize>,
    pub confs: Vec<f32>,
    pub iou: Option<f32>,
    pub names: Option<Vec<String>>,  // names
    /*pub context_length: Option<usize>,
    pub min_width: Option<f32>,
    pub min_height: Option<f32>,
    pub unclip_ratio: f32, // DB*/
    pub yolo_version: Option<YoloVersion>,
    pub yolo_preds: Option<YoloPreds>,
}

impl Default for ConfigOrt {
    fn default() -> Self {
        Self {
            onnx_path: String::new(),
            ort_lib_path: String::new(),
            device: DeviceType::CPU,
            profile: false,
            batch_size: 1,
            iiixs: vec![],
            num_dry_run: 3,

            trt_engine_cache_enable: true,
            trt_int8_enable: false,
            trt_fp16_enable: false,

            nc: Some(80),
            confs: vec![0.3f32],
            iou: None,
            names: None,
            /*context_length: None,
            min_width: None,
            min_height: None,
            unclip_ratio: 1.5,*/
            yolo_version: None,
            yolo_preds: None,
        }
    }
}

#[allow(dead_code)]
impl ConfigOrt {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_model(mut self, onnx_path: &str) -> Result<Self> {
        self.onnx_path = onnx_path.to_string();
        Ok(self)
    }

    pub fn with_ort_lib_path(mut self, ort_lib_path: &str) -> Result<Self> {
        self.ort_lib_path = ort_lib_path.to_string();
        Ok(self)
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn with_batch(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn with_dry_run(mut self, n: usize) -> Self {
        self.num_dry_run = n;
        self
    }

    pub fn with_device(mut self, device_type: DeviceType) -> Self {
        self.device = device_type;
        self
    }

    pub fn with_trt_fp16(mut self, x: bool) -> Self {
        self.trt_fp16_enable = x;
        self
    }

    pub fn with_yolo_version(mut self, x: YoloVersion) -> Self {
        self.yolo_version = Some(x);
        self
    }

    pub fn with_profile(mut self, profile: bool) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_names(mut self, names: &[&str]) -> Self {
        self.names = Some(names.iter().map(|x| x.to_string()).collect::<Vec<String>>());
        self
    }

/*    pub fn with_min_width(mut self, x: f32) -> Self {
        self.min_width = Some(x);
        self
    }

    pub fn with_min_height(mut self, x: f32) -> Self {
        self.min_height = Some(x);
        self
    }*/

    pub fn with_yolo_preds(mut self, x: YoloPreds) -> Self {
        self.yolo_preds = Some(x);
        self
    }

    pub fn with_nc(mut self, nc: usize) -> Self {
        self.nc = Some(nc);
        self
    }

    pub fn with_iou(mut self, x: f32) -> Self {
        self.iou = Some(x);
        self
    }

    pub fn with_confs(mut self, x: &[f32]) -> Self {
        self.confs = x.to_vec();
        self
    }

    pub fn with_ixx(mut self, i: usize, ii: usize, x: MinOptMax) -> Self {
        self.iiixs.push(Iiix::from((i, ii, x)));
        self
    }
}
