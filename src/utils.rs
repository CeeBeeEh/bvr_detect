use std::{fs, io};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

pub(crate) fn file_to_vec(filename: String) -> io::Result<Vec<String>> {
    let file_in = fs::File::open(filename)?;
    let file_reader = BufReader::new(file_in);
    Ok(file_reader.lines().filter_map(io::Result::ok).collect())
}

pub(crate) fn trace(is_test: bool, l_type: &str, l_step: &str, detect: Instant, _detect_elapsed: Duration) -> Duration {
    if is_test {
        println!("{} | Total={}ms | {}={:.2?}", l_type, detect.elapsed().as_millis(), l_step, detect.elapsed() - _detect_elapsed);
    }
    else {
        log::trace!("{} | Total={:.2?} | {}={:.2?}", l_type, detect.elapsed(), l_step, detect.elapsed() - _detect_elapsed);
    }
    detect.elapsed()
}