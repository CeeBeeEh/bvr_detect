pub trait Nms {
    fn iou(&self, other: &Self) -> f32;
    fn confidence(&self) -> f32;
}
