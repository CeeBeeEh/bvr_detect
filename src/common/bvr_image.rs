use image::{DynamicImage, GenericImageView};
use crate::data::LabelThreshold;

#[derive(Debug, Clone)]
pub struct BvrImage {
    pub image: DynamicImage,
    pub img_width: i32,
    pub img_height: i32,
    pub threshold: f32,
    pub augment: bool,
    pub wanted_labels: Option<Vec<u16>>,
}

impl BvrImage {
    pub fn new(image: DynamicImage, threshold: f32, augment: bool, label_filters: Option<Vec<u16>>) -> Self {
        let (img_width, img_height) = image.dimensions();
        Self {
            image,
            img_width: img_width as i32,
            img_height: img_height as i32,
            threshold,
            augment,
            wanted_labels: label_filters
        }
    }

    pub fn get_ratio(&self) -> f32 {
        self.img_width as f32 / self.img_height as f32
    }
    
    pub fn is_label_wanted(&self, comp_id: u16) -> bool {
        match &self.wanted_labels {
            Some(masks) => {
                masks.contains(&comp_id)       
            },
            None => { false }
        }        
    }
}