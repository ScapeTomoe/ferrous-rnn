use crate::activation::Activation;
use crate::loss::Loss;
use crate::params::RnnGrads;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;

pub struct Rnn {
    wx: Array2<f32>,
    wh: Array2<f32>,
    b: Array1<f32>,
    hidden_dim: usize,
    activation: Activation,
}

impl Rnn {
    pub fn new(input_dim: usize, hidden_dim: usize, activation: Activation) -> Self {
        Rnn {
            wx: Array2::zeros((hidden_dim, input_dim)),
            wh: Array2::zeros((hidden_dim, hidden_dim)),
            b: Array1::zeros(hidden_dim),
            hidden_dim,
            activation,
        }
    }

    pub fn update(&mut self, grads: &RnnGrads, lr: f32) {
        self.wx = &self.wx - lr * &grads.d_wx;
        self.wh = &self.wh - lr * &grads.d_wh;
        self.b = &self.b - lr * &grads.d_b;
    }

    // 推論用（速い、メモリ少ない）
    pub fn forward(&self, x: &Array2<f32>, mode: OutputMode) -> RnnOutput {
        let mut h = Array1::zeros(self.hidden_dim);
        let mut history = vec![];

        for t in 0..x.nrows() {
            let x_t = x.row(t);
            let z = self.wx.dot(&x_t) + self.wh.dot(&h) + &self.b;
            h = z.mapv(|v| self.activation.apply(v));
            history.push(h.clone());
        }

        match mode {
            OutputMode::Sequences => {
                let mut result = Array2::zeros((history.len(), self.hidden_dim));
for (t, h) in history.iter().enumerate() {
    result.row_mut(t).assign(h);
}
RnnOutput::Sequences(result)
            }
            OutputMode::LastOnly => RnnOutput::LastOnly(h),
        }
    }
    // 学習用（historyも返す）
    pub fn forward_train(&self, x: &Array2<f32>) -> (RnnOutput, Vec<Array1<f32>>) {
        let mut h = Array1::zeros(self.hidden_dim);
        let mut history = vec![];

        for t in 0..x.nrows() {
            let x_t = x.row(t);
            let z = self.wx.dot(&x_t) + self.wh.dot(&h) + &self.b;
            h = z.mapv(|v| self.activation.apply(v));
            history.push(h.clone());
        }

        let mut result = Array2::zeros((history.len(), self.hidden_dim));
        for (t, h) in history.iter().enumerate() {
            result.row_mut(t).assign(h);
        }
        let sequences = RnnOutput::Sequences(result); 

        (sequences, history)
    }
    pub fn backward(
        &self,
        x: &Array2<f32>,
        history: &[Array1<f32>],
        delta: &Array2<f32>,
    ) -> RnnGrads {
        let seq_len = x.nrows();
        let mut d_wx = Array2::zeros(self.wx.dim());
        let mut d_wh = Array2::zeros(self.wh.dim());
        let mut d_b = Array1::zeros(self.b.dim());
        let mut d_h = Array1::zeros(self.hidden_dim);

        for t in (0..seq_len).rev() {
            let h_prev = if t == 0 {
                Array1::zeros(self.hidden_dim)
            } else {
                history[t - 1].clone()
            };

            // δt・f'(zt)
            let d_raw = (&delta.row(t) + &d_h) * history[t].mapv(|v| self.activation.grad(v));

            // 勾配の累積
            d_wx = d_wx
                + d_raw
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&x.row(t).insert_axis(Axis(0)));
            d_wh = d_wh
                + d_raw
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&h_prev.view().insert_axis(Axis(0)));
            d_b = d_b + &d_raw;

            // 前ステップへの誤差伝播
            d_h = self.wh.t().dot(&d_raw);
        }

        RnnGrads { d_wx, d_wh, d_b }
    }

    pub fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &Array2<f32>, //正解
        lr: f32,         //学習率
        epochs: usize,   //学習回数
        loss: &Loss,     //損失関数
    ) {
        for epoch in 0..epochs {
            let (output, history) = self.forward_train(x);

            let y_pred = match output {
                RnnOutput::Sequences(arr) => arr,
                RnnOutput::LastOnly(_) => panic!("fit requires Sequences mode"),
            };

            let loss_val = loss.apply(&y_pred, y);
            let delta = loss.grad(&y_pred, y);
            let grads = self.backward(x, &history, &delta);
            self.update(&grads, lr);

            if epoch % 100 == 0 {
                println!("epoch: {},loss:{}", epoch, loss_val);
            }
        }
    }
}

pub enum RnnOutput {
    Sequences(Array2<f32>), // (seq_len, hidden_dim)
    LastOnly(Array1<f32>),  // (hidden_dim,)
}

pub enum OutputMode {
    Sequences,
    LastOnly,
}
