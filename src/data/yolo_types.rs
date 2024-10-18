//! File/code adapted from https://github.com/jamjamjon/usls

use ndarray::{ArrayBase, ArrayView, Axis, Dim, IxDyn, IxDynImpl, ViewRepr};

// Possibly change this to be a generic model type, instead of Yolo specific
#[derive(Debug, Copy, Clone, Default)]
pub enum YoloVersion {
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    #[default] V11,
    RTDETR,
}

impl YoloVersion {
    pub fn name(&self) -> String {
        match self {
            Self::V5 => "v5".to_string(),
            Self::V6 => "v6".to_string(),
            Self::V7 => "v7".to_string(),
            Self::V8 => "v8".to_string(),
            Self::V9 => "v9".to_string(),
            Self::V10 => "v10".to_string(),
            Self::V11 => "v11".to_string(),
            Self::RTDETR => "rtdetr".to_string(),
        }
    }

    pub fn from(version: String) -> YoloVersion {
        match version.to_lowercase().as_str() {
            "v5" => YoloVersion::V5,
            "v6" => YoloVersion::V6,
            "v7" => YoloVersion::V7,
            "v8" => YoloVersion::V8,
            "v9" => YoloVersion::V9,
            "v10" => YoloVersion::V10,
            "v11" => YoloVersion::V11,
            "rtdetr" => YoloVersion::RTDETR,
            _ => YoloVersion::V11,
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
