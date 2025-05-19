//! File/code adapted from https://github.com/jamjamjon/usls

use ndarray::{ArrayBase, ArrayView, Axis, Dim, IxDyn, IxDynImpl, ViewRepr};

// Possibly change this to be a generic model type, instead of Yolo specific
#[derive(Debug, Copy, Clone, Default)]
pub enum ModelVersion {
    YoloV5,
    YoloV6,
    YoloV7,
    YoloV8,
    YoloV9,
    YoloV10,
    #[default] YoloV11,
    YoloV12,
}

impl ModelVersion {
    pub fn name(&self) -> String {
        match self {
            Self::YoloV5 => "YoloV5".to_string(),
            Self::YoloV6 => "YoloV6".to_string(),
            Self::YoloV7 => "YoloV7".to_string(),
            Self::YoloV8 => "YoloV8".to_string(),
            Self::YoloV9 => "YoloV9".to_string(),
            Self::YoloV10 => "YoloV10".to_string(),
            Self::YoloV11 => "YoloV11".to_string(),
            Self::YoloV12 => "YoloV12".to_string(),
        }
    }

    pub fn from(version: String) -> ModelVersion {
        match version.to_lowercase().as_str() {
            "yolov5" => ModelVersion::YoloV5,
            "yolov6" => ModelVersion::YoloV6,
            "yolov7" => ModelVersion::YoloV7,
            "yolov8" => ModelVersion::YoloV8,
            "yolov9" => ModelVersion::YoloV9,
            "yolov10" => ModelVersion::YoloV10,
            "yolov11" => ModelVersion::YoloV11,
            "yolov12" => ModelVersion::YoloV12,
            _ => ModelVersion::YoloV11,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum BoxType {
    /// 1
    Cxcywh,

    /// 2 Cxcybr
    Cxcyxy,

    /// 3 Tlbr
    Xyxy,

    /// 4  Tlwh
    Xywh,

    /// 5  Tlcxcy
    XyCxcy,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum ClssType {
    Clss,
    ConfCls,
    ClsConf,
    ConfClss,
    ClssConf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnchorsPosition {
    Before,
    After,
}

#[derive(Debug, Clone, PartialEq)]
pub struct YoloPreds {
    pub clss: ClssType,
    pub bbox: Option<BoxType>,
    pub anchors: Option<AnchorsPosition>,
    pub is_bbox_normalized: bool,
    pub apply_nms: bool,
    pub apply_softmax: bool,
}

impl Default for YoloPreds {
    fn default() -> Self {
        Self {
            clss: ClssType::Clss,
            bbox: None,
            anchors: None,
            is_bbox_normalized: false,
            apply_nms: true,
            apply_softmax: false,
        }
    }
}

#[allow(dead_code)]
impl YoloPreds {
    pub fn apply_nms(mut self, x: bool) -> Self {
        self.apply_nms = x;
        self
    }

    pub fn apply_softmax(mut self, x: bool) -> Self {
        self.apply_softmax = x;
        self
    }

    pub fn n_clss() -> Self {
        // Classification: NClss
        Self {
            clss: ClssType::Clss,
            ..Default::default()
        }
    }

    pub fn n_a_cxcywh_confclss() -> Self {
        // YOLOv5 | YOLOv6 | YOLOv7 | YOLOX : NACxcywhConfClss
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::ConfClss,
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_a_cxcywh_confclss_coefs() -> Self {
        // YOLOv5 Segment : NACxcywhConfClssCoefs
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::ConfClss,
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_a() -> Self {
        // YOLOv8 | YOLOv9 : NCxcywhClssA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn n_a_xyxy_confcls() -> Self {
        // YOLOv10 : NAXyxyConfCls
        Self {
            bbox: Some(BoxType::Xyxy),
            clss: ClssType::ConfCls,
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_a_cxcywh_clss_n() -> Self {
        // RTDETR
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            anchors: Some(AnchorsPosition::Before),
            is_bbox_normalized: true,
            ..Default::default()
        }
    }

    pub fn box_type(&self) -> Option<&BoxType> {
        self.bbox.as_ref()
    }

    pub fn is_anchors_first(&self) -> bool {
        matches!(self.anchors, Some(AnchorsPosition::Before))
    }

    pub fn is_cls_type(&self) -> bool {
        matches!(self.clss, ClssType::ClsConf | ClssType::ConfCls)
    }

    pub fn is_clss_type(&self) -> bool {
        matches!(
            self.clss,
            ClssType::ClssConf | ClssType::ConfClss | ClssType::Clss
        )
    }

    pub fn is_conf_at_end(&self) -> bool {
        matches!(self.clss, ClssType::ClssConf | ClssType::ClsConf)
    }

    pub fn is_conf_independent(&self) -> bool {
        !matches!(self.clss, ClssType::Clss)
    }

    #[allow(clippy::type_complexity)]
    pub fn parse_preds<'a>(
        &'a self,
        x: ArrayBase<ViewRepr<&'a f32>, Dim<IxDynImpl>>,
        nc: usize,
    ) -> (
        Option<ArrayView<f32, IxDyn>>,
        Option<ArrayView<f32, IxDyn>>,
        ArrayView<f32, IxDyn>,
        Option<ArrayView<f32, IxDyn>>,
    ) {
        let x = if self.is_anchors_first() {
            x
        } else {
            x.reversed_axes()
        };

        // get each tasks slices
        let (slice_bboxes, _xs) = x.split_at(Axis(1), 4);

        let (slice_id, slice_clss, slice_confs, _xs) = match self.clss {
            ClssType::ConfClss => {
                let (confs, _xs) = _xs.split_at(Axis(1), 1);
                let (clss, _xs) = _xs.split_at(Axis(1), nc);
                (None, clss, Some(confs), _xs)
            }
            ClssType::ClssConf => {
                let (clss, _xs) = _xs.split_at(Axis(1), nc);
                let (confs, _xs) = _xs.split_at(Axis(1), 1);
                (None, clss, Some(confs), _xs)
            }
            ClssType::ConfCls => {
                let (clss, _xs) = _xs.split_at(Axis(1), 1);
                let (ids, _xs) = _xs.split_at(Axis(1), 1);
                (Some(ids), clss, None, _xs)
            }
            ClssType::ClsConf => {
                let (ids, _xs) = _xs.split_at(Axis(1), 1);
                let (clss, _xs) = _xs.split_at(Axis(1), 1);
                (Some(ids), clss, None, _xs)
            }
            ClssType::Clss => {
                let (clss, _xs) = _xs.split_at(Axis(1), nc);
                (None, clss, None, _xs)
            }
        };

        (
            Some(slice_bboxes),
            slice_id,
            slice_clss,
            slice_confs,
        )
    }
}
