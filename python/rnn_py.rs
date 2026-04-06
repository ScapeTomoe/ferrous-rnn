use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, ToPyArray};
use crate::rnn::{Rnn, OutputMode, RnnOutput};
use crate::activation::Activation;
use crate::loss::Loss; // 損失関数をインポート

#[pyclass]
struct PyRnn {
    inner: Rnn,
}

#[pymethods]
impl PyRnn {
    #[new]
    fn new(input_dim: usize, hidden_dim: usize, activation: &str) -> PyResult<Self> {
        // 文字列から活性化関数を選択
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

    // 学習用メソッド
    fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray2<f32>, lr: f32, epochs: usize, loss_type: &str) {
        let x_rust = x.to_owned_array();
        let y_rust = y.to_owned_array();
        let loss = match loss_type.to_lowercase().as_str() {
            "mse" => Loss::Mse,
            _ => Loss::CrossEntropy,
        };
        self.inner.fit(&x_rust, &y_rust, lr, epochs, &loss);
    }

    // 推論用メソッド
    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> &'py PyArray2<f32> {
        let x_rust = x.to_owned_array();
        let output = self.inner.forward(&x_rust, OutputMode::Sequences);
        
        match output {
            RnnOutput::Sequences(arr) => arr.to_pyarray(py),
            RnnOutput::LastOnly(arr) => {
                // LastOnly の場合は 2次元化して返す
                arr.insert_axis(ndarray::Axis(0)).to_pyarray(py)
            }
        }
    }
}

#[pymodule]
fn ferrous_rnn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRnn>()?;
    Ok(())
}