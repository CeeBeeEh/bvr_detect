//! File adapted from: https://github.com/jamjamjon
//!
//! Functions to preprocess images.

use rayon::iter::ParallelIterator;
use anyhow::{bail, Result};
use fast_image_resize::{
    images::{CroppedImageMut, Image as FirImage},
    pixels::PixelType,
    ResizeAlg, ResizeOptions, Resizer,
};
use image::RgbImage;
use ndarray::{Array, IxDyn};
use rayon::prelude::IntoParallelRefIterator;
use crate::detection_runners::input_wrapper::X;

/// Resize mode enum.
pub enum ResizeMode {
    FitExact,
    Letterbox,
}

/// Main preprocessing entry point.
pub fn preprocess(
    xs: &[FirImage],
    target_h: u32,
    target_w: u32,
    resize_mode: ResizeMode,
) -> Result<Vec<X>> {
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Nearest);

    // Parallel preprocessing per image
    let image_tensors: Vec<Vec<f32>> = xs
        .par_iter()
        .map(|img| {
            let mut resizer = Resizer::new();
            let resized = match resize_mode {
                ResizeMode::FitExact => resize_image(img, target_h, target_w, &mut resizer, &options)?,
                ResizeMode::Letterbox => letterbox_image(img, target_h, target_w, 114, false, &mut resizer, &options)?,
            };
            nchw_normalize_flat(&resized)
        })
        .collect::<Result<_>>()?;

    // Stack into one batch manually
    let channels = 3;
    let height = target_h as usize;
    let width = target_w as usize;
    let image_size = channels * height * width;

    let mut batch_flat: Vec<f32> = Vec::with_capacity(xs.len() * image_size);
    for img in image_tensors.iter() {
        batch_flat.extend_from_slice(img);
    }

    let batch = Array::from_shape_vec(
        (xs.len(), channels, height, width),
        batch_flat,
    )?.into_dyn();

    Ok(vec![X::from(batch)])
}

pub fn make_divisible(x: usize, divisor: usize) -> usize {
    // (x + divisor - 1) / divisor * divisor
    x.div_ceil(divisor) * divisor
}

pub fn to_fir_image<'a>(mut image: RgbImage) -> FirImage<'a> {
    let (width, height) = image.dimensions();
    let buffer = std::mem::take(&mut image).into_raw();

    FirImage::from_vec_u8(width, height, buffer, PixelType::U8x3)
        .expect("Failed to convert to FirImage")
}

fn resize_image<'a>(
    img: &FirImage,
    target_h: u32,
    target_w: u32,
    resizer: &mut Resizer,
    config: &ResizeOptions,
) -> Result<FirImage<'a>> {
/*    if img.width() == target_w && img.height() == target_h {
        return Ok(img);
    }*/

    let mut dst = FirImage::new(target_w, target_h, PixelType::U8x3);
    resizer.resize(img, &mut dst, config)?;
    Ok(dst)
}

fn letterbox_image<'a>(
    img: &FirImage,
    target_h: u32,
    target_w: u32,
    bg: u8,
    center: bool,
    resizer: &mut Resizer,
    resize_options: &ResizeOptions,
) -> Result<FirImage<'a>> {
    let (w0, h0) = (img.width(), img.height());
    let scale = (target_w as f32 / w0 as f32).min(target_h as f32 / h0 as f32);
    let new_w = (w0 as f32 * scale).round() as u32;
    let new_h = (h0 as f32 * scale).round() as u32;

    let mut padded = FirImage::from_vec_u8(
        target_w,
        target_h,
        vec![bg; (target_w * target_h * 3) as usize],
        PixelType::U8x3,
    )?;

    let (left, top) = if center {
        ((target_w - new_w) / 2, (target_h - new_h) / 2)
    } else {
        (0, 0)
    };

    let mut cropped = CroppedImageMut::new(&mut padded, left, top, new_w, new_h)?;
    resizer.resize(img, &mut cropped, resize_options)?;

    Ok(padded)
}

fn normalize_image(img: &FirImage) -> Result<Array<f32, IxDyn>> {
    let buf = img.buffer();
    if buf.len() != (img.width() * img.height() * 3) as usize {
        bail!("Unexpected buffer size: got {}, expected {}", buf.len(), img.width() * img.height() * 3);
    }

    let float_data: Vec<f32> = buf.iter().map(|&v| v as f32 / 255.0).collect();
    let array = Array::from_shape_vec((img.height() as usize, img.width() as usize, 3), float_data)?
        .into_dyn();

    Ok(array)
}

fn to_nchw(array: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
    if array.ndim() != 3 {
        bail!("Expected 3D tensor (HWC), got ndim={}", array.ndim());
    }

    let shape = array.shape();
    let shape_view = array.view();
    let chw = shape_view
        .to_shape((shape[0], shape[1], shape[2]))?
        .permuted_axes([2, 0, 1]);

    Ok(chw.to_owned().into_dyn())
}

fn nchw_normalize_flat(img: &FirImage) -> Result<Vec<f32>> {
    let buf = img.buffer();
    let w = img.width() as usize;
    let h = img.height() as usize;

    if buf.len() != w * h * 3 {
        bail!("Unexpected buffer size: got {}, expected {}", buf.len(), w * h * 3);
    }

    let mut out = vec![0.0f32; buf.len()];
    let (_c, hw) = (3, w * h);

    for i in 0..hw {
        let r = buf[3 * i];
        let g = buf[3 * i + 1];
        let b = buf[3 * i + 2];

        out[i] = r as f32 / 255.0;             // Channel 0
        out[i + hw] = g as f32 / 255.0;         // Channel 1
        out[i + 2 * hw] = b as f32 / 255.0;     // Channel 2
    }

    Ok(out)
}