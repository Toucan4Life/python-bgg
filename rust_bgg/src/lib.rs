use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use std::fs::File;
use std::io::{BufWriter, Write};

#[inline]
fn xlogx(x: f64) -> f64 {
    if x > 0.0 {
        x * x.ln()
    } else {
        0.0
    }
}

#[inline]
fn llr(a: f64, b: f64, c: f64, d: f64) -> f64 {
    2.0 * (
        xlogx(a) + xlogx(b) + xlogx(c) + xlogx(d)
        - xlogx(a + b) - xlogx(c + d)
        - xlogx(a + c) - xlogx(b + d)
        + xlogx(a + b + c + d)
    )
}

#[inline]
fn process_edge(
    i: i32,
    j: i32,
    a: f64,
    pop_i: f64,
    pop_j: f64,
    n: f64,
    min_a: f64,
    llr_thresh: f64,
) -> Option<(i32, i32, f64, f64, f64)> {
    if i >= j {
        return None;
    }

    if a < min_a {
        return None;
    }

    let b = pop_i - a;
    let c = pop_j - a;
    let d = n - a - b - c;

    let score = llr(a, b, c, d);
    if score < llr_thresh {
        return None;
    }

    if pop_i <= 0.0 || pop_j <= 0.0 {
        return None;
    }

    let cosine = a / (pop_i * pop_j).sqrt();
    let jaccard = a / (pop_i + pop_j - a);

    Some((i, j, a, cosine, jaccard))
}

#[pyfunction]
fn process_edges<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f64>,
    popularity: PyReadonlyArray1<f64>,
    n: f64,
    min_a: f64,
    llr_thresh: f64,
) -> PyResult<(
    &'py PyArray1<i32>,
    &'py PyArray1<i32>,
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
)> {
    let rows = rows.as_slice()?;
    let cols = cols.as_slice()?;
    let data = data.as_slice()?;
    let pop = popularity.as_slice()?;

    let mut result_i = Vec::new();
    let mut result_j = Vec::new();
    let mut result_a = Vec::new();
    let mut result_cosine = Vec::new();
    let mut result_jaccard = Vec::new();

    for k in 0..data.len() {
        let i = rows[k];
        let j = cols[k];
        let a = data[k];
        
        if let Some((ei, ej, ea, cosine, jaccard)) = process_edge(
            i,
            j,
            a,
            pop[i as usize],
            pop[j as usize],
            n,
            min_a,
            llr_thresh,
        ) {
            result_i.push(ei);
            result_j.push(ej);
            result_a.push(ea);
            result_cosine.push(cosine);
            result_jaccard.push(jaccard);
        }
    }

    Ok((
        PyArray1::from_vec(py, result_i),
        PyArray1::from_vec(py, result_j),
        PyArray1::from_vec(py, result_a),
        PyArray1::from_vec(py, result_cosine),
        PyArray1::from_vec(py, result_jaccard),
    ))
}

#[pyfunction]
fn process_edges_to_csv(
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f64>,
    popularity: PyReadonlyArray1<f64>,
    n: f64,
    min_a: f64,
    llr_thresh: f64,
    output_path: String,
) -> PyResult<usize> {
    let rows = rows.as_slice()?;
    let cols = cols.as_slice()?;
    let data = data.as_slice()?;
    let pop = popularity.as_slice()?;

    let file = File::create(&output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create file: {}", e)))?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "game_i;game_j;a;cosine;jaccard")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write header: {}", e)))?;

    let mut count = 0usize;

    for k in 0..data.len() {
        let i = rows[k];
        let j = cols[k];
        let a = data[k];
        
        if let Some((ei, ej, ea, cosine, jaccard)) = process_edge(
            i,
            j,
            a,
            pop[i as usize],
            pop[j as usize],
            n,
            min_a,
            llr_thresh,
        ) {
            writeln!(writer, "{};{};{};{};{}", ei, ej, ea, cosine, jaccard)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write row: {}", e)))?;
            count += 1;
        }
    }

    writer.flush()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to flush writer: {}", e)))?;

    Ok(count)
}

#[pymodule]
fn rust_bgg(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_edges, m)?)?;
    m.add_function(wrap_pyfunction!(process_edges_to_csv, m)?)?;
    Ok(())
}
