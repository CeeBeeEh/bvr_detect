use image::{DynamicImage, GenericImageView};

#[derive(Debug, Clone)]
pub struct BvrImage {
    pub image: DynamicImage,
    pub img_width: i32,
    pub img_height: i32,
    pub threshold: f32,
    pub augment: bool,
}

impl BvrImage {
    pub fn new(image: DynamicImage, threshold: f32, augment: bool) -> Self {
        let (img_width, img_height) = image.dimensions();
        Self {
            image,
            img_width: img_width as i32,
            img_height: img_height as i32,
            threshold,
            augment,
        }
    }

    pub fn get_ratio(&self) -> f32 {
        self.img_width as f32 / self.img_height as f32
    }
}