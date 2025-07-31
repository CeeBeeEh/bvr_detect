use crate::data::{ConfigOrt, Xs, Y};

pub trait InferenceProcess: Sized {
    type Input<'a>; // DynamicImage
    type Thresholds;

    /// Creates a new instance of the model with the given options.
    fn new(options: ConfigOrt) -> anyhow::Result<Self>;

    /// Pre-process the input data.
    fn preprocess<'a>(&self, xs: &[Self::Input<'a>]) -> anyhow::Result<Xs>;

    /// Executes the model on the preprocessed data.
    fn inference(&mut self, xs: Xs) -> anyhow::Result<Xs>;

    /// Post-process the model's output.
    fn postprocess<'a>(&self, xs: Xs, xs0: &[Self::Input<'a>], thresh: &[Self::Thresholds]) -> anyhow::Result<Vec<Y>>;

    /// Executes the full pipeline.
    fn run<'a>(&mut self, xs: &[Self::Input<'a>], thresh: &[Self::Thresholds], profile: bool) -> anyhow::Result<Vec<Y>> {
        let t_pre = std::time::Instant::now();
        let ys = self.preprocess(xs)?;
        let t_pre = t_pre.elapsed();

        let t_exe = std::time::Instant::now();
        let ys = self.inference(ys)?;
        let t_exe = t_exe.elapsed();

        let t_post = std::time::Instant::now();
        let ys = self.postprocess(ys, xs, thresh)?;
        let t_post = t_post.elapsed();

        if profile {
            log::info!("> Preprocess: {t_pre:?} | Inference: {t_exe:?} | Postprocess: {t_post:?}");
        }

        println!("> Preprocess: {t_pre:?} | Inference: {t_exe:?} | Postprocess: {t_post:?}");

        Ok(ys)
    }

    #[allow(dead_code)]
    fn print_time(&self);
}
