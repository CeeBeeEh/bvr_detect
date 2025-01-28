use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, PartialOrd)]
pub struct BvrBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub w: f32,
    pub h: f32,

    pub conf: f32,
}

impl BvrBox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            w: x2 - x1,
            h: y2 - y1,
            conf,
        }
    }

    /// Returns the width of the bounding box.
    pub fn width(&self) -> f32 {
        self.w
    }

    /// Returns the height of the bounding box.
    pub fn height(&self) -> f32 {
        self.h
    }

    /// Returns the minimum x-coordinate of the bounding box.
    pub fn x_min(&self) -> f32 {
        self.x1
    }

    /// The minimum y-coordinate of the bounding box.
    pub fn y_min(&self) -> f32 {
        self.y1
    }

    /// Returns the maximum x-coordinate of the bounding box.
    pub fn x_max(&self) -> f32 {
        self.x1 + self.w
    }

    /// The maximum x-coordinate of the bounding box.
    pub fn y_max(&self) -> f32 {
        self.y1 + self.h
    }

    /// Returns the center x-coordinate of the bounding box.
    pub fn cx(&self) -> f32 {
        self.x1 + self.w / 2.
    }

    /// Returns the center y-coordinate of the bounding box.
    pub fn cy(&self) -> f32 {
        self.y1 + self.h / 2.
    }

    /// Returns the bounding box coordinates as `(x1, y1, x2, y2)`.
    pub fn xy1_xy2(&self) -> (f32, f32, f32, f32) {
        (self.x1, self.y1, self.x2, self.y2)
    }

    /// Returns the bounding box coordinates and size as `(x, y, w, h)`.
    pub fn xy1_wh(&self) -> (f32, f32, f32, f32) {
        (self.x1, self.y1, self.w, self.h)
    }

    /// Returns the center coordinates and size of the bounding box as `(cx, cy, w, h)`.
    pub fn cxy_wh(&self) -> (f32, f32, f32, f32) {
        (self.cx(), self.cy(), self.w, self.h)
    }

    /// Computes the area of the bounding box.
    pub fn area(&self) -> f32 {
        self.h * self.w
    }

    /// Computes the perimeter of the bounding box.
    pub fn perimeter(&self) -> f32 {
        (self.h + self.w) * 2.
    }

    /// Computes the intersection area between this bounding box and another.
    pub fn intersect(&self, other: &BvrBox) -> f32 {
        let left = self.x1.max(other.x1);
        let right = (self.x1 + self.w).min(other.x1 + other.w);
        let top = self.y1.max(other.y1);
        let bottom = (self.y1 + self.h).min(other.y1 + other.h);
        (right - left).max(0.) * (bottom - top).max(0.)
    }

    /// Computes the union area between this bounding box and another.
    pub fn union(&self, other: &BvrBox) -> f32 {
        self.area() + other.area() - self.intersect(other)
    }

    // /// Computes the intersection over union (IoU) between this bounding box and another.
    // pub fn iou(&self, other: &Bbox) -> f32 {
    //     self.intersect(other) / self.union(other)
    // }

    /// Checks if this bounding box completely contains another bounding box `other`.
    pub fn contains(&self, other: &BvrBox) -> bool {
        self.x_min() <= other.x_min()
            && self.x_max() >= other.x_max()
            && self.y_min() <= other.y_min()
            && self.y_max() >= other.y_max()
    }

    pub fn scale(&mut self, factor: f32) {
        self.w = self.w * factor;
        self.h = self.h * factor;
        self.x2 = self.x1 + self.w;
        self.y2 = self.y1 + self.h;
    }

    pub fn as_xy_wh_i32(&self) -> (i32, i32, i32, i32) {
        (self.x1.round() as i32,
         self.y1.round() as i32,
         self.w.round() as i32,
         self.h.round() as i32)
    }

    pub fn as_xy_wh_f32_rounded(&self) -> (f32, f32, f32, f32) {
        (self.x1.round(),
         self.y1.round(),
         self.w.round(),
         self.h.round())
    }

    pub fn as_x1y1_x2y2_i32(&self) -> (i32, i32, i32, i32) {
        (self.x1.round() as i32,
         self.y1.round() as i32,
         self.x2.round() as i32,
         self.y2.round() as i32)
    }

    pub fn as_x1y1_x2y2_f32_rounded(&self) -> (f32, f32, f32, f32) {
        (self.x1.round(),
         self.y1.round(),
         self.x2.round(),
         self.y2.round())
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
    /// A `BvrBox` instance with updated coordinates and dimensions.
    pub fn with_x1y1_x2y2(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        self.x1 = x1;
        self.y1 = y1;
        self.x2 = x2;
        self.y2 = y2;

        self.w = x2 - x1;
        self.h = y2 - y1;
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
    /// A `BvrBox` instance with updated coordinates and dimensions.
    pub fn with_x1y1_wh(mut self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.x1 = x;
        self.y1 = y;
        self.w = w;
        self.h = h;

        self.x2 = x + w;
        self.y2 = y + h;
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
    /// A `BvrBox` instance with updated coordinates and dimensions.
    pub fn with_cxcy_wh(mut self, cx: f32, cy: f32, w: f32, h: f32) -> Self {
        self.x1 = cx - (w / 2.0);
        self.y1 = cy - (h / 2.0);
        self.w = w;
        self.h = h;

        self.x2 = cx + (w / 2.0);
        self.y2 = cy + (h / 2.0);
        self
    }


    pub fn with_confidence(mut self, x: f32) -> Self {
        self.conf = x;
        self
    }
}