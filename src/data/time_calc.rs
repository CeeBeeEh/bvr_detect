//! File/code adapted from https://github.com/jamjamjon/usls

use std::time::Duration;

#[derive(Debug, Default)]
pub struct TimeCalc {
    n: usize,
    duration: Vec<Duration>,
}

#[allow(dead_code)]
impl TimeCalc {
    pub fn total(&self) -> Duration {
        self.duration.iter().sum::<Duration>()
    }

    pub fn n(&self) -> usize {
        self.n / self.duration.len()
    }

    pub fn avg(&self) -> Duration {
        self.total() / self.n() as u32
    }

    pub fn avg_i(&self, i: usize) -> Duration {
        if i >= self.duration.len() {
            panic!("Index out of bound");
        }
        self.duration[i] / self.n() as u32
    }

    pub fn ts(&self) -> &Vec<Duration> {
        &self.duration
    }

    pub fn add_or_push(&mut self, i: usize, x: Duration) {
        match self.duration.get_mut(i) {
            Some(elem) => *elem += x,
            None => {
                if i >= self.duration.len() {
                    self.duration.push(x)
                }
            }
        }
        self.n += 1;
    }

    pub fn clear(&mut self) {
        self.n = Default::default();
        self.duration = Default::default();
    }
}
