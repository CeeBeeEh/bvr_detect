//! File/code adapted from https://github.com/jamjamjon/usls

use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Axis};
use rayon::prelude::*;
use regex::Regex;

use crate::common::{BoxType, BvrDetection, ModelVersion, YoloPreds};
use crate::data::{ConfigOrt, DynConf, ImageOps, MinOptMax, Xs, X, Y};
use crate::detection_runners::inference_process::InferenceProcess;
use crate::detection_runners::ort_detector::OrtEngine;

#[derive(Debug)]
pub struct OrtYOLO {
    engine: OrtEngine,
    nc: usize,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    confs: DynConf,
    iou: f32,
    names: Vec<String>,
    layout: YoloPreds,
    version: Option<ModelVersion>,
}

impl InferenceProcess for OrtYOLO {
    type Input = DynamicImage;

    fn new(options: ConfigOrt) -> Result<Self> {

        let engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );

        // YOLO Outputs Format
        let (version, layout) = match options.yolo_version {
            Some(ver) => match ver {
                ModelVersion::YoloV5 | ModelVersion::YoloV6 | ModelVersion::YoloV7 => (Some(ver), YoloPreds::n_a_cxcywh_confclss()),
                ModelVersion::YoloV8 | ModelVersion::YoloV9 | ModelVersion::YoloV11 => (Some(ver), YoloPreds::n_cxcywh_clss_a()),
                ModelVersion::YoloV10 => (Some(ver), YoloPreds::n_a_xyxy_confcls().apply_nms(false)),
            },
            None => match options.yolo_preds {
                None => anyhow::bail!("No clear YOLO version or YOLO Format specified."),
                Some(fmt) => (None, fmt),
            }
        };

        // Class names: user-defined.or(parsed)
        let names_parsed = Self::fetch_names(&engine);
        let names = match names_parsed {
            Some(names_parsed) => match options.names {
                Some(names) => {
                    if names.len() == names_parsed.len() {
                        Some(names)
                    } else {
                        anyhow::bail!(
                            "The lengths of parsed class names: {} and user-defined class names: {} do not match.",
                            names_parsed.len(),
                            names.len(),
                        );
                    }
                }
                None => Some(names_parsed),
            },
            None => options.names,
        };

        // nc: names.len().or(options.nc)
        let nc = match &names {
            Some(names) => names.len(),
            None => match options.nc {
                Some(nc) => nc,
                None => anyhow::bail!(
                    "Unable to obtain the number of classes. Please specify them explicitly using `options.with_nc(usize)` or `options.with_names(&[&str])`."
                ),
            }
        };

        // Class names
        let names = match names {
            None => Self::n2s(nc),
            Some(names) => names,
        };

        // Confs & Iou
        let confs = DynConf::new(&options.confs, nc);
        let iou = options.iou.unwrap_or(0.45);

        // Summary
        log::info!("YOLO Version: {:?}", version);

        Ok(Self {
            engine,
            confs,
            iou,
            nc,
            height,
            width,
            batch,
            names,
            layout,
            version,
        })
    }

    fn preprocess(&self, xs: &[Self::Input]) -> Result<Xs> {
        let xs_ = X::apply(&[
                ImageOps::Letterbox(
                    xs,
                    self.height() as u32,
                    self.width() as u32,
                    "CatmullRom",
                    114,
                    "auto",
                    false,
                ),
                ImageOps::Normalize(0., 255.),
                ImageOps::Nhwc2nchw,
            ])?;
        Ok(Xs::from(xs_))
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    fn postprocess(&self, xs: Xs, xs0: &[Self::Input]) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, preds)| {
                let mut y = Y::default();

                // parse preditions
                let (
                    slice_bboxes,
                    slice_id,
                    slice_clss,
                    slice_confs,
                ) = self.layout.parse_preds(preds, self.nc);

                let image_width = xs0[idx].width() as f32;
                let image_height = xs0[idx].height() as f32;
                let ratio =
                    (self.width() as f32 / image_width).min(self.height() as f32 / image_height);

                let y_bboxes = slice_bboxes?
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, bbox)| {

                        // confidence & class_id
                        let (class_id, confidence) = match &slice_id {
                            Some(ids) => (ids[[i, 0]] as _, slice_clss[[i, 0]] as _),
                            None => {
                                let (class_id, &confidence) = slice_clss
                                    .slice(s![i, ..])
                                    .into_iter()
                                    .enumerate()
                                    .max_by(|a, b| a.1.total_cmp(b.1))?;

                                match &slice_confs {
                                    None => (class_id, confidence),
                                    Some(slice_confs) => {
                                        (class_id, confidence * slice_confs[[i, 0]])
                                    }
                                }
                            }
                        };

                        // filtering low scores
                        if confidence < self.confs[class_id] {
                            return None;
                        }

                        // Bounding boxes
                        let bbox = bbox.mapv(|x| x / ratio);

                        let bbox = if self.layout.is_bbox_normalized {
                            (
                                bbox[0] * self.width() as f32,
                                bbox[1] * self.height() as f32,
                                bbox[2] * self.width() as f32,
                                bbox[3] * self.height() as f32,
                            )
                        } else {
                            (bbox[0], bbox[1], bbox[2], bbox[3])
                        };

                        let (_cx, _cy, x, y, w, h) = match self.layout.box_type()? {
                            BoxType::Cxcywh => {
                                let (cx, cy, w, h) = bbox;
                                let x = (cx - w / 2.).max(0.);
                                let y = (cy - h / 2.).max(0.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Xyxy => {
                                let (x, y, x2, y2) = bbox;
                                let (w, h) = (x2 - x, y2 - y);
                                let (cx, cy) = ((x + x2) / 2., (y + y2) / 2.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Xywh => {
                                let (x, y, w, h) = bbox;
                                let (cx, cy) = (x + w / 2., y + h / 2.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Cxcyxy => {
                                let (cx, cy, x2, y2) = bbox;
                                let (w, h) = ((x2 - cx) * 2., (y2 - cy) * 2.);
                                let x = (x2 - w).max(0.);
                                let y = (y2 - h).max(0.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::XyCxcy => {
                                let (x, y, cx, cy) = bbox;
                                let (w, h) = ((cx - x) * 2., (cy - y) * 2.);
                                (cx, cy, x, y, w, h)
                            }
                        };

                        // filtering unreliably small objects
                        if w < 15.0 || h < 15.0 {
                            return None;
                        }

                        let y_bbox = BvrDetection::default()
                                    .with_x1y1_wh(x, y, w, h)
                                    .with_confidence(confidence)
                                    .with_class_id(class_id as isize)
                                    .with_label(&self.names[class_id]);

                        Some(y_bbox)
                    })
                    .collect::<Vec<_>>();

                // Bboxes
                if !y_bboxes.is_empty() {
                    y = y.with_detections(&y_bboxes);
                    if self.layout.apply_nms {
                        y = y.apply_nms(self.iou);
                    }
                }

                Some(y)
            })
            .collect();

        Ok(ys)
    }

    fn print_time(&self) {
        println!("Avg: {:?}", self.engine.infer_time.avg());
    }
}

#[allow(dead_code)]
impl OrtYOLO {
    pub fn batch(&self) -> usize {
        self.batch.opt()
    }

    pub fn width(&self) -> usize {
        self.width.opt()
    }

    pub fn height(&self) -> usize {
        self.height.opt()
    }

    pub fn version(&self) -> Option<&ModelVersion> {
        self.version.as_ref()
    }


    pub fn layout(&self) -> &YoloPreds {
        &self.layout
    }

    fn fetch_names(engine: &OrtEngine) -> Option<Vec<String>> {
        // fetch class names from onnx metadata
        // String format: `{0: 'person', 1: 'bicycle', 2: 'sports ball', ..., 27: "yellow_lady's_slipper"}`
        engine.try_fetch("names").map(|names| {
            let re = Regex::new(r#"(['"])([-()\w '"]+)(['"])"#).unwrap();
            let mut names_ = vec![];
            for (_, [_, name, _]) in re.captures_iter(&names).map(|x| x.extract()) {
                names_.push(name.to_string());
            }
            names_
        })
    }

    fn n2s(n: usize) -> Vec<String> {
        (0..n).map(|x| format!("# {}", x)).collect::<Vec<String>>()
    }
}
