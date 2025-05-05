use image::{GenericImageView};

#[derive(Debug, Clone)]
pub struct BvrImage {
    pub image: image::DynamicImage,
    pub img_width: u32,
    pub img_height: u32,
    pub threshold: f32,
    pub augment: bool,
    pub wanted_labels: Option<Vec<u16>>,
}

impl BvrImage {
    pub fn new(image: image::DynamicImage, threshold: f32, augment: bool, label_filters: Option<Vec<u16>>) -> Self {
        let (img_width, img_height) = image.dimensions();
        Self {
            image,
            img_width,
            img_height,
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
    
    pub fn clone_image(&self) -> image::DynamicImage {
        self.image.clone()
    }

    pub fn get_img_width(&self) -> u32 {
        self.img_width
    }

    pub fn get_img_height(&self) -> u32 {
        self.img_height
    }
    
    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }

    pub fn get_wanted_labels(&self) -> Option<&Vec<u16>> {
        self.wanted_labels.as_ref()
    }
}