/*pub mod ffi {
    use std::ffi::{c_char, c_double, c_int, c_uchar, CStr, CString};
    use std::slice;
    use image::{DynamicImage, ImageBuffer, Rgb};
    use tokio::runtime::Runtime;
    use crate::bvr_detect::{detect, init_detector, BvrImage};
    use crate::detection::BvrDetection;

    #[repr(C)]
    pub struct BvrBoxFfi {
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        width: i32,
        height: i32,
    }

    #[repr(C)]
    pub struct BvrDetectionFfi {
        id: u32,
        class_id: u32,
        track_id: u32,
        bbox: BvrBoxFfi,
        label: *const c_char,
        last_inference_time: u128,
        inference_hits: u32,
        frame_hits: u32,
        confidence: f32,
        num_matches: u32,
        camera_name: *const c_char,
        camera_id: u32,
    }

    #[repr(C)]
    pub struct BvrImageFfi {
//        pub image_data: *const c_uchar, // Pointer to the image data (e.g., pixel buffer)
        pub image_len: usize,           // Length of the image data (in bytes)
        pub img_width: c_int,           // Image width
        pub img_height: c_int,          // Image height
        pub conf_thres: c_double,       // Confidence threshold
        pub iou_thres: c_double,        // IoU threshold
        pub augment: bool,              // Augmentation flag
    }

    // Conversion function to create a BvrImageFfi from a DynamicImage
    impl BvrImageFfi {
        pub fn from_dynamic_image(image: DynamicImage, conf_thres: f64, iou_thres: f64, augment: bool) -> Self {
            let (img_width, img_height) = (image.width() as i32, image.height() as i32);
            let image_buffer = image.to_rgb8().into_raw(); // Get the raw bytes from the image

            BvrImageFfi {
//                image_data: image_buffer.as_ptr(),
                image_len: image_buffer.len(),
                img_width,
                img_height,
                conf_thres,
                iou_thres,
                augment,
            }
        }
    }

    // Helper function to convert Rust's BvrDetection to FfiBvrDetection
    fn to_ffi_bvr_detection(detection: BvrDetection) -> BvrDetectionFfi {
        BvrDetectionFfi {
            id: detection.id,
            class_id: detection.class_id,
            track_id: detection.track_id,
            bbox: BvrBoxFfi {
                x1: detection.bbox.x1,
                y1: detection.bbox.y1,
                x2: detection.bbox.x2,
                y2: detection.bbox.y2,
                width: detection.bbox.width,
                height: detection.bbox.height,
            },
            label: CString::new(detection.label).unwrap().into_raw(),
            last_inference_time: detection.last_inference_time,
            inference_hits: detection.inference_hits,
            frame_hits: detection.frame_hits,
            confidence: detection.confidence,
            num_matches: detection.num_matches,
            camera_name: CString::new(detection.camera_name).unwrap().into_raw(),
            camera_id: detection.camera_id,
        }
    }

    #[no_mangle]
    pub extern "C" fn create_bvr_image_ffi(
        image_len: usize,
        img_width: c_int,
        img_height: c_int,
        conf_thres: c_double,
        iou_thres: c_double,
        augment: bool,
    ) -> *mut BvrImageFfi {
        let bvr_image_ffi = BvrImageFfi {
            image_len,
            img_width,
            img_height,
            conf_thres,
            iou_thres,
            augment,
        };

        // Allocate memory for the struct and return a pointer
        Box::into_raw(Box::new(bvr_image_ffi))
    }

    // Free a BvrImageFfi struct
    #[no_mangle]
    pub extern "C" fn free_bvr_image_ffi(bvr_image_ffi: *mut BvrImageFfi) {
        if !bvr_image_ffi.is_null() {
            unsafe {
                let _ = Box::from_raw(bvr_image_ffi); // Automatically deallocated when Box is dropped
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn is_running_ffi() -> bool {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let is_running = crate::bvr_detect::is_running().await;
            is_running.unwrap()
        })
    }

    #[no_mangle]
    pub extern "C" fn init_detector_ffi(model_path: *const c_char, classes_path: *const c_char) {
        // Create a Tokio runtime to handle async execution
        let rt = Runtime::new().unwrap();

        // Convert the C strings into Rust strings
        let model_path_str = unsafe { CStr::from_ptr(model_path).to_str().unwrap() }.to_string();
        let classes_path_str = unsafe { CStr::from_ptr(classes_path).to_str().unwrap() }.to_string();

        // Block on the async function
        rt.block_on(async {
            init_detector(model_path_str, classes_path_str, 640, 640).await;
        });
    }

    #[no_mangle]
    pub extern "C" fn detect_ffi(image_data: *const c_uchar, bvr_image_ffi: BvrImageFfi, out_detections: *mut *mut BvrDetectionFfi, out_len: *mut usize,
    ) -> c_int {
        let rt = Runtime::new().unwrap();

        let result = rt.block_on(async {

            let img_slice = unsafe {
                slice::from_raw_parts(image_data, (bvr_image_ffi.img_width * bvr_image_ffi.img_height * 3) as usize)
            };

            // Create an ImageBuffer from raw data
            let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                bvr_image_ffi.img_width as u32,
                bvr_image_ffi.img_height as u32,
                img_slice.to_vec()
            ).expect("Invalid image buffer");

            // Convert ImageBuffer to DynamicImage
            let image = DynamicImage::ImageRgb8(img_buffer);

            //let image = DynamicImage::from(image_data);
            let bvr_image = BvrImage {
                image,
                img_width: bvr_image_ffi.img_width,
                img_height: bvr_image_ffi.img_height,
                conf_thres: bvr_image_ffi.conf_thres,
                iou_thres: bvr_image_ffi.iou_thres,
                augment: bvr_image_ffi.augment,
            };
            detect(bvr_image).await
        });

        match result {
            Ok(detections) => {
                let len = detections.len();
                let mut ffi_detections: Vec<BvrDetectionFfi> = detections
                    .into_iter()
                    .map(to_ffi_bvr_detection)
                    .collect();

                // Allocate memory for the C-compatible detections array
                let detections_ptr = ffi_detections.as_mut_ptr();
                std::mem::forget(ffi_detections); // Prevent Rust from dropping the Vec

                unsafe {
                    *out_detections = detections_ptr;
                    *out_len = len;
                }

                0 // Return 0 on success
            }
            Err(_) => {
                -1 // Return -1 on error
            }
        }
    }
}*/