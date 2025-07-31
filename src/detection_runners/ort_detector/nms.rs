use bvr_common::BvrDetection;

pub trait Nms {
    fn iou(&self, other: &Self) -> f32;
    fn confidence(&self) -> f32;
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