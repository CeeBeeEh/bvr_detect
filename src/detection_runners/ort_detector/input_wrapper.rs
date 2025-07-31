//! File/code adapted from https://github.com/jamjamjon/usls

use anyhow::Result;
use ndarray::{Array, IxDyn};

/// Model input, wrapper over [`Array<f32, IxDyn>`]
#[derive(Debug, Clone, Default)]
pub struct X(pub Array<f32, IxDyn>);

impl From<Array<f32, IxDyn>> for X {
    fn from(x: Array<f32, IxDyn>) -> Self {
        Self(x)
    }
}

impl From<Vec<f32>> for X {
    fn from(x: Vec<f32>) -> Self {
        Self(Array::from_vec(x).into_dyn().into_owned())
    }
}

impl std::ops::Deref for X {
    type Target = Array<f32, IxDyn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl X {

    pub fn from_shape_vec(shape: &[usize], xs: Vec<f32>) -> Result<Self> {
        Ok(Self::from(Array::from_shape_vec(shape, xs)?))
    }

    pub fn ndim(&self) -> usize {
        self.0.ndim()
    }
}