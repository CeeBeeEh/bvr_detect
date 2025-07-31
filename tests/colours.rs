use image::Rgb;

pub(crate) fn get_class_colour(class: usize) -> Rgb<u8> {
    match class {
        0 => Rgb([128, 0, 128]),     // purple (people)
        1..=8 => Rgb([0, 255, 0]),   // green (vehicles)
        14..=23 => Rgb([255, 0, 0]), // red (animals
        _ => Rgb([0, 0, 255])        // blue (everything else)
    }
}