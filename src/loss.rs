use ndarray::Array2;

pub enum Loss {
    Mse,
    CrossEntropy,
}

pub fn mse(y_pred: &Array2<f32>, y_test: &Array2<f32>) -> f32 {
    let diff = y_pred - y_test;
    0.5 * (&diff * &diff).mean().unwrap()
}

pub fn crossentropy(y_pred: &Array2<f32>, y_test: &Array2<f32>) -> f32 {
    const EPSILON: f32 = 1e-7;
    let y_clamped = y_pred.mapv(|v| v.clamp(EPSILON, 1.0 - EPSILON));
    (-y_test * y_clamped.mapv(f32::ln)).mean().unwrap()
}

pub fn mse_grad(y_pred: &Array2<f32>, y_test: &Array2<f32>) -> Array2<f32> {
    let n = y_pred.len() as f32;
    2.0 / n * (y_pred - y_test)
}

pub fn crossentropy_grad(y_pred: &Array2<f32>, y_test: &Array2<f32>) -> Array2<f32> {
    const EPSILON: f32 = 1e-7;
    let y_clamped = y_pred.mapv(|v| v.clamp(EPSILON, 1.0 - EPSILON));
    -y_test / y_clamped
}

impl Loss {
    pub fn apply(&self, y_pred: &Array2<f32>, y: &Array2<f32>) -> f32 {
        match self {
            Loss::Mse => mse(y_pred, y),
            Loss::CrossEntropy => crossentropy(y_pred, y),
        }
    }

    pub fn grad(&self, y_pred: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
        match self {
            Loss::Mse => mse_grad(y_pred, y),
            Loss::CrossEntropy => crossentropy_grad(y_pred, y),
        }
    }
}
