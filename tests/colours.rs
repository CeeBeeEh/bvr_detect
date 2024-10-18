use image::Rgba;

pub(crate) fn get_class_colour(class: usize) -> Rgba<u8> {
    match class {
        0 => Rgba([128, 0, 128, 255]),     // purple (people)
        1..=8 => Rgba([0, 255, 0, 255]),   // green (vehicles)
        14..=23 => Rgba([255, 0, 0, 255]), // red (animals
        _ => Rgba([0, 0, 255, 255])        // blue (everything else)
    }
}