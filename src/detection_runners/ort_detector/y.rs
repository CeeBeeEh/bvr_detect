//! File/code adapted from https://github.com/jamjamjon/usls

use crate::data::BvrDetection;
use crate::detection_runners::ort_detector::nms::Nms;
use crate::detection_runners::ort_detector::prob::Prob;

/// Container for inference results for each image.
///
/// This struct possible outputs from an image inference process.
///
/// # Fields
///
/// * `probs` - Optionally contains the probability scores for the detected objects.
/// * `bboxes` - Optionally contains a vector of BvrDetection.
#[derive(Clone, PartialEq, Default)]
pub struct Y {
    probs: Option<Prob>,
    detections: Option<Vec<BvrDetection>>,
}

impl std::fmt::Debug for Y {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Y");
        if let Some(x) = &self.probs {
            f.field("Probabilities", &x);
        }
        if let Some(x) = &self.detections {
            if !x.is_empty() {
                f.field("BvrDetections", &x);
            }
        }
        f.finish()
    }
}

impl Y {
    /// Sets the `probs` field with the provided probability scores.
    ///
    /// # Arguments
    ///
    /// * `probs` - A reference to a `Prob` instance to be cloned and set in the struct.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new probabilities set.
    pub fn with_probs(mut self, probs: &Prob) -> Self {
        self.probs = Some(probs.clone());
        self
    }

    /// Sets the `bboxes` field with the provided vector of bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `bboxes` - A slice of `BvrBox` to be set.
    ///
    /// # Returns
    ///
    /// * `Self` - The updated struct instance with the new detections.
    pub fn with_detections(mut self, detections: &[BvrDetection]) -> Self {
        self.detections = Some(detections.to_vec());
        self
    }

    /// Returns a reference to the `probs` field, if it exists.
    ///
    /// # Returns
    ///
    /// * `Option<&Prob>` - A reference to the probabilities, or `None` if it is not set.
    pub fn probs(&self) -> Option<&Prob> {
        self.probs.as_ref()
    }

    ///
    /// # Returns
    ///
    /// * `Option<&Vec<BvrDetection>>` - A reference to the vector of detections, or `None` if it is not set.
    pub fn detections(&self) -> Option<&Vec<BvrDetection>> {
        self.detections.as_ref()
    }

    pub fn apply_nms(mut self, iou_threshold: f32) -> Self {
        match &mut self.detections {
            None =>  { self },
            Some(ref mut bboxes) => {
                Self::nms(bboxes, iou_threshold);
                self
            }
        }
    }

    pub fn nms<T: Nms>(boxes: &mut Vec<T>, iou_threshold: f32) {
        boxes.sort_by(|b1, b2| {
            b2.confidence()
                .partial_cmp(&b1.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut current_index = 0;
        for index in 0..boxes.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = boxes[prev_index].iou(&boxes[index]);
                if iou > iou_threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                boxes.swap(current_index, index);
                current_index += 1;
            }
        }
        boxes.truncate(current_index);
    }
}
