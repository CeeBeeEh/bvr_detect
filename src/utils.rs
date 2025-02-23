use std::{fs, io};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};
use rand::{rng, Rng};
use rand::distr::Alphanumeric;

#[allow(dead_code)]
pub(crate) fn trace(is_test: bool, l_type: &str, l_step: &str, detect: Instant, _detect_elapsed: Duration) -> Duration {
    if is_test {
        println!("{} | Total={}ms | {}={:.2?}", l_type, detect.elapsed().as_millis(), l_step, detect.elapsed() - _detect_elapsed);
    }
    else {
        log::trace!("{} | Total={:.2?} | {}={:.2?}", l_type, detect.elapsed(), l_step, detect.elapsed() - _detect_elapsed);
    }
    detect.elapsed()
}

#[allow(dead_code)]
pub(crate) fn string_random(n: usize) -> String {
    rng()
        .sample_iter(&Alphanumeric)
        .take(n)
        .map(char::from)
        .collect()
}

pub fn human_bytes(size: f64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    let mut size = size;
    let mut unit_index = 0;
    let k = 1024.;

    while size >= k && unit_index < units.len() - 1 {
        size /= k;
        unit_index += 1;
    }

    format!("{:.1} {}", size, units[unit_index])
}