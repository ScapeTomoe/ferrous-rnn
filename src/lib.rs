pub mod activation;
pub mod rnn;
pub mod params;
pub mod loss;

use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, ToPyArray};
use numpy::PyArrayMethods; 
use crate::rnn::{Rnn, OutputMode, RnnOutput};
use crate::activation::Activation;
use crate::loss::Loss;

#[pyclass]
struct PyRnn {
    inner: Rnn,
}

#[pymethods]
impl PyRnn {
    #[new]
    fn new(input_dim: usize, hidden_dim: usize, activation: &str) -> PyResult<Self> {
        let act = match activation.to_lowercase().as_str() {
            "relu" => Activation::Relu,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "leaky_relu" => Activation::LeakyRelu,
            "elu" => Activation::Elu,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid activation type")),
        };
        Ok(PyRnn { inner: Rnn::new(input_dim, hidden_dim, act) })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray2<f32>, lr: f32, epochs: usize, loss_type: &str) {
        let x_rust = x.to_owned_array();
        let y_rust = y.to_owned_array();
        let loss = match loss_type.to_lowercase().as_str() {
            "mse" => Loss::Mse,
            _ => Loss::CrossEntropy,
        };
        self.inner.fit(&x_rust, &y_rust, lr, epochs, &loss);
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> Bound<'py, PyArray2<f32>> {
    let x_rust = x.to_owned_array();
    let output = self.inner.forward(&x_rust, OutputMode::Sequences);
    
    match output {
        RnnOutput::Sequences(arr) => arr.to_pyarray_bound(py),
        RnnOutput::LastOnly(arr) => {
            arr.insert_axis(ndarray::Axis(0)).to_pyarray_bound(py)
        }
    }
}
}

// モジュール名は Cargo.toml の [lib] name と一致させる必要があります
#[pymodule]
fn ferrous_rnn(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRnn>()?;
    Ok(())
}