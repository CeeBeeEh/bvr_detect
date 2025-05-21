use std::time::Instant;
use crate::data::{ConfigOrt, Xs, Y};
use crate::utils;

pub trait InferenceProcess: Sized {
    type Input; // DynamicImage
    type Thresholds;

    /// Creates a new instance of the model with the given options.
    fn new(options: ConfigOrt) -> anyhow::Result<Self>;

    /// Pre-process the input data.
    fn preprocess(&self, xs: &[Self::Input]) -> anyhow::Result<Xs>;

    /// Executes the model on the preprocessed data.
    fn inference(&mut self, xs: Xs) -> anyhow::Result<Xs>;

    /// Post-process the model's output.
    fn postprocess(&self, xs: Xs, xs0: &[Self::Input], thresh: &[Self::Thresholds]) -> anyhow::Result<Vec<Y>>;

    /// Executes the full pipeline.
    fn run(&mut self, xs: &[Self::Input], thresh: &[Self::Thresholds]) -> anyhow::Result<Vec<Y>> {
        let ys = self.preprocess(xs)?;
        let ys = self.inference(ys)?;
        let ys = self.postprocess(ys, xs, thresh)?;
        Ok(ys)
    }

    /// Executes the full pipeline.
    fn forward(&mut self, xs: &[Self::Input], thresh: &[Self::Thresholds], profile: bool) -> anyhow::Result<Vec<Y>> {
        let detect_time =  Instant::now();

        let t_pre = std::time::Instant::now();
        let ys = self.preprocess(xs)?;
        let t_pre = t_pre.elapsed();

        let mut _detect_elapsed = detect_time.elapsed();
        _detect_elapsed = utils::trace(true, "TIME", "Preprocessing input", detect_time, _detect_elapsed);

        let t_exe = std::time::Instant::now();
        let ys = self.inference(ys)?;
        let t_exe = t_exe.elapsed();

        _detect_elapsed = utils::trace(true, "TIME", "Detection run", detect_time, _detect_elapsed);

        let t_post = std::time::Instant::now();
        let ys = self.postprocess(ys, xs, thresh)?;
        let t_post = t_post.elapsed();

        _detect_elapsed = utils::trace(true, "TIME", "Postprocessing", detect_time, _detect_elapsed);

        if profile {
            log::info!("> Preprocess: {t_pre:?} | Inference: {t_exe:?} | Postprocess: {t_post:?}");
        }

        Ok(ys)
    }

    #[allow(dead_code)]
    fn print_time(&self);
}
