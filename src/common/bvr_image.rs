use aksr::Builder;
use fast_image_resize::images::{Image as FirImage};
use fast_image_resize::PixelType;
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage, SubImage};

#[derive(Debug, Clone, Default)]
pub struct BvrImage {
    pub image: RgbImage,
    pub img_width: u32,
    pub img_height: u32,
    pub threshold: f32,
    pub pad: bool,
    pub wanted_labels: Option<Vec<u16>>,
}

#[derive(Builder, Debug, Clone, Default)]
pub struct ImageTransformInfo {
    pub width_src: u32,
    pub height_src: u32,
    pub width_dst: u32,
    pub height_dst: u32,
    pub height_scale: f32,
    pub width_scale: f32,
    pub height_pad: f32,
    pub width_pad: f32,
}

impl std::ops::Deref for BvrImage {
    type Target = RgbImage;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl std::ops::DerefMut for BvrImage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.image
    }
}

impl From<DynamicImage> for BvrImage {
    fn from(image: DynamicImage) -> Self {
        Self {
            image: image.to_rgb8(),
            ..Default::default()
        }
    }
}

impl From<GrayImage> for BvrImage {
    fn from(image: GrayImage) -> Self {
        Self {
            image: DynamicImage::from(image).to_rgb8(),
            ..Default::default()
        }
    }
}

impl From<RgbImage> for BvrImage {
    fn from(image: RgbImage) -> Self {
        Self {
            image,
            ..Default::default()
        }
    }
}

impl From<RgbaImage> for BvrImage {
    fn from(image: RgbaImage) -> Self {
        Self {
            image: DynamicImage::from(image).to_rgb8(),
            ..Default::default()
        }
    }
}

impl<I> From<SubImage<I>> for BvrImage
where
    I: std::ops::Deref,
    I::Target: image::GenericImageView<Pixel = image::Rgb<u8>> + 'static,
{
    fn from(sub_image: SubImage<I>) -> Self {
        let image: RgbImage = sub_image.to_image();

        Self {
            image,
            ..Default::default()
        }
    }
}

impl From<BvrImage> for DynamicImage {
    fn from(image: BvrImage) -> Self {
        image.into_dyn()
    }
}

impl From<BvrImage> for RgbImage {
    fn from(image: BvrImage) -> Self {
        image.into_rgb8()
    }
}

impl BvrImage {
    pub fn new(image: RgbImage, threshold: f32, pad: bool, label_filters: Option<Vec<u16>>) -> Self {
        let (img_width, img_height) = image.dimensions();
        Self {
            image,
            img_width,
            img_height,
            threshold,
            pad,
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
    
    pub fn clone_image(&self) -> RgbImage {
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

    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }

    pub fn height(&self) -> u32 {
        self.image.height()
    }

    pub fn width(&self) -> u32 {
        self.image.width()
    }

    pub fn size(&self) -> u32 {
        self.image.as_raw().len() as u32
    }

    pub fn to_u32s(&self) -> Vec<u32> {
        use rayon::prelude::*;

        self.image
            .as_raw()
            .par_chunks(3)
            .map(|c| ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32))
            .collect()
    }

    pub fn to_f32s(&self) -> Vec<f32> {
        use rayon::prelude::*;

        self.image
            .as_raw()
            .into_par_iter()
            .map(|x| *x as f32)
            .collect()
    }

    pub fn to_dyn(&self) -> DynamicImage {
        DynamicImage::from(self.image.clone())
    }

    pub fn to_rgb8(&self) -> RgbImage {
        self.image.clone()
    }

    pub fn take_as_fir_image(&mut self) -> FirImage {
        let (width, height) = self.image.dimensions();
        let buffer = std::mem::take(&mut self.image).into_raw();

        FirImage::from_vec_u8(width, height, buffer, PixelType::U8x3)
            .expect("Failed to convert to FirImage")
    }

    pub fn into_dyn(self) -> DynamicImage {
        DynamicImage::from(self.image)
    }

    pub fn into_rgb8(self) -> RgbImage {
        self.image
    }
}