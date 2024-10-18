use std::{fs, io};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};
use rand::{distributions::Alphanumeric, thread_rng, Rng};

#[allow(dead_code)]
pub(crate) fn file_to_vec(filename: String) -> io::Result<Vec<String>> {
    let file_in = fs::File::open(filename)?;
    let file_reader = BufReader::new(file_in);
    Ok(file_reader.lines().filter_map(io::Result::ok).collect())
}

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
    thread_rng()
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