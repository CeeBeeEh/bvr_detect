use serde::{Deserialize, Serialize};
use crate::common::BvrBox;
use crate::detection_runners::ort_detector::nms::Nms;

#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct BvrDetection {
    pub id: isize,
    pub class_id: isize,
    pub track_id: isize,
    pub bbox: BvrBox,
    pub label: Option<String>,
    pub last_inference_time: u128,
    pub inference_hits: u32,
    pub frame_hits: u32,
    pub confidence: f32,
    pub num_matches: u32,
    pub camera_name: Option<String>,
    pub camera_id: isize,
}

impl Nms for BvrDetection {
    /// Computes the intersection over union (IoU) between this bounding box and another.
    fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }

    /// Returns the confidence score of the bounding box.
    fn confidence(&self) -> f32 {
        self.confidence
    }
}

impl BvrDetection {
    pub fn new(class_id: isize, bbox: BvrBox, label: Option<String>, confidence: f32) -> Self {
        Self {
            id: 0,
            class_id,
            track_id: 0,
            bbox,
            label,
            last_inference_time: 0,
            inference_hits: 0,
            frame_hits: 0,
            confidence,
            num_matches: 0,
            camera_name: None,
            camera_id: 0,
        }
    }

    /// Sets the bounding box's coordinates using `(x1, y1, x2, y2)` and calculates width and height.
    ///
    /// # Arguments
    ///
    /// * `x1` - The x-coordinate of the top-left corner.
    /// * `y1` - The y-coordinate of the top-left corner.
    /// * `x2` - The x-coordinate of the bottom-right corner.
    /// * `y2` - The y-coordinate of the bottom-right corner.
    ///
    /// # Returns
    ///
    /// A `BvrDetection` instance with updated coordinates and dimensions.
    pub fn with_x1y1_x2y2(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        self.bbox = BvrBox::default().with_x1y1_x2y2(x1, y1, x2, y2);
        self
    }

    /// Sets the bounding box's coordinates and dimensions using `(x, y, w, h)`.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the top-left corner.
    /// * `y` - The y-coordinate of the top-left corner.
    /// * `w` - The width of the bounding box.
    /// * `h` - The height of the bounding box.
    ///
    /// # Returns
    ///
    /// A `BvrDetection` instance with updated coordinates and dimensions.
    pub fn with_x1y1_wh(mut self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.bbox = BvrBox::default().with_x1y1_wh(x, y, w, h);
        self
    }

    /// Sets the bounding box's coordinates and dimensions using `(cx, cy, w, h)`.
    ///
    /// # Arguments
    ///
    /// * `cx` - The x-coordinate of the horizontal center.
    /// * `cy` - The y-coordinate of the vertical center.
    /// * `w` - The width of the bounding box.
    /// * `h` - The height of the bounding box.
    ///
    /// # Returns
    ///
    /// A `BvrDetection` instance with updated coordinates and dimensions.
    pub fn with_cxcy_wh(mut self, cx: f32, cy: f32, w: f32, h: f32) -> Self {
        self.bbox = BvrBox::default().with_x1y1_wh(cx, cy, w, h);
        self
    }

    /// Sets the confidence score of the detection_runners.
    ///
    /// # Arguments
    ///
    /// * `conf` - The confidence score to be set.
    ///
    /// # Returns
    ///
    /// A `BvrDetection` instance with updated confidence score.
    pub fn with_confidence(mut self, conf: f32) -> Self {
        self.confidence = conf;
        self
    }

    /// Sets the class ID of the detection_runners.
    ///
    /// # Arguments
    ///
    /// * `class_id` - The class ID to be set.
    ///
    /// # Returns
    ///
    /// A `BvrDetection` instance with updated class ID.
    pub fn with_class_id(mut self, class_id: isize) -> Self {
        self.class_id = class_id;
        self
    }

    /// Sets the optional name of the detection_runners.
    ///
    /// # Arguments
    ///
    /// * `label` - The name to be set.
    ///
    /// # Returns
    ///
    /// A `BvrDetection` instance with updated name.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn get_label(&self) -> String {
        self.label.clone().unwrap_or("Unknown".to_string())
    }

    pub fn inference_hit(&mut self) {
        self.inference_hits += 1;
    }

    pub fn frame_hit(&mut self) {
        self.inference_hits += 1;
    }

    /// Computes the intersection area between this detection_runners and another.
    pub fn intersect(&self, other: &BvrDetection) -> f32 {
        self.bbox.intersect(&other.bbox)
    }

    /// Computes the union area between this bounding box and another.
    pub fn union(&self, other: &BvrDetection) -> f32 {
        self.bbox.union(&other.bbox)
    }

    pub fn print_detection(&self) {
        println!(
            "Detection: ID: {}, Class: {}, BBox: {:?}, Confidence: {:.2}",
            self.id, self.class_id, self.bbox, self.confidence
        );
    }
}