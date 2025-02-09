//! File/code adapted from https://github.com/jamjamjon/usls

use anyhow::Result;
use std::collections::HashMap;
use std::ops::{Deref, Index};

use crate::detection_runners::ort_detector::input_wrapper::X;
use crate::utils::string_random;

#[derive(Debug, Default, Clone)]
pub struct Xs {
    map: HashMap<String, X>,
    names: Vec<String>,
}

impl From<X> for Xs {
    fn from(x: X) -> Self {
        let mut xs = Self::default();
        xs.push(x);
        xs
    }
}

impl From<Vec<X>> for Xs {
    fn from(xs: Vec<X>) -> Self {
        let mut ys = Self::default();
        for x in xs {
            ys.push(x);
        }
        ys
    }
}

impl Xs {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn push(&mut self, value: X) {
        loop {
            let key = string_random(5);
            if !self.map.contains_key(&key) {
                self.names.push(key.to_string());
                self.map.insert(key.to_string(), value);
                break;
            }
        }
    }

    pub fn push_kv(&mut self, key: &str, value: X) -> Result<()> {
        if !self.map.contains_key(key) {
            self.names.push(key.to_string());
            self.map.insert(key.to_string(), value);
            Ok(())
        } else {
            anyhow::bail!("Xs already contains key: {:?}", key)
        }
    }

    pub fn names(&self) -> &Vec<String> {
        &self.names
    }
}

impl Deref for Xs {
    type Target = HashMap<String, X>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl Index<&str> for Xs {
    type Output = X;

    fn index(&self, index: &str) -> &Self::Output {
        self.map.get(index).expect("Index was not found in `Xs`")
    }
}

impl Index<usize> for Xs {
    type Output = X;

    fn index(&self, index: usize) -> &Self::Output {
        self.names
            .get(index)
            .and_then(|key| self.map.get(key))
            .expect("Index was not found in `Xs`")
    }
}

pub struct XsIter<'a> {
    inner: std::vec::IntoIter<&'a X>,
}

impl<'a> Iterator for XsIter<'a> {
    type Item = &'a X;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a> IntoIterator for &'a Xs {
    type Item = &'a X;
    type IntoIter = XsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let values: Vec<&X> = self.names.iter().map(|x| &self.map[x]).collect();
        XsIter {
            inner: values.into_iter(),
        }
    }
}
