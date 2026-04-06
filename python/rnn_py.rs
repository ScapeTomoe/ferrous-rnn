use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, PyArray1};
use crate::rnn::{Rnn, OutputMode};
use crate::activation::Activation;

#[pyclass]
struct PyRnn(Rnn);

#[pymethods]
impl PyRnn {
    #[new]
    fn new(input_dim: usize, hidden_dim: usize, activation: &str) -> PyResult<Self> {
        let act = Activation::from_str(activation)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(PyRnn(Rnn::new(input_dim, hidden_dim, act)))
    }

    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        sequences: bool,
    ) -> PyResult<&'py PyArray2<f32>> {
        let x = x.as_array().to_owned();
        let mode = if sequences { OutputMode::Sequences } else { OutputMode::LastOnly };
        // ...
    }
}

#[pymodule]
fn rnn_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRnn>()?;
    Ok(())
}