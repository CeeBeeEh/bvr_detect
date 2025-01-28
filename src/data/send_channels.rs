use crate::common::{BvrDetection, BvrImage};

#[derive(Debug)]
pub struct DetectionState {
    pub opt_rx: crossbeam_channel::Receiver<Box<BvrImage>>,
    pub det_tx: crossbeam_channel::Sender<Box<Vec<BvrDetection>>>,
}

#[derive(Debug)]
pub struct SendState {
    pub opt_tx: crossbeam_channel::Sender<Box<BvrImage>>,
    pub det_rx: crossbeam_channel::Receiver<Box<Vec<BvrDetection>>>,
}