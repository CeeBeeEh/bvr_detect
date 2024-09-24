use std::sync::mpsc::{Receiver, Sender};
use crate::bvr_data::{BvrDetection, BvrImage};

#[derive(Debug)]
pub struct DetectionState {
    pub opt_rx: Receiver<BvrImage>,
    pub det_tx: Sender<Vec<BvrDetection>>,
}

#[derive(Debug)]
pub struct SendState {
    pub opt_tx: Sender<BvrImage>,
    pub det_rx: Receiver<Vec<BvrDetection>>,
}