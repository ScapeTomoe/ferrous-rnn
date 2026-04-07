use crate::activation::Activation;
use crate::loss::Loss;
use crate::params::RnnGrads;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use rand::thread_rng;

pub struct Rnn {
    wx: Array2<f32>,
    wh: Array2<f32>,
    b: Array1<f32>,
    hidden_dim: usize,
    activation: Activation,
}

impl Rnn {
    pub fn new(input_dim: usize, hidden_dim: usize, activation: Activation) -> Self {
        let std_dev=(2.0/(input_dim+hidden_dim) as f32).sqrt();
        let dist=Normal::new(0.0,std_dev).unwrap();
        Rnn {
            wx: Array2::random((hidden_dim, input_dim),dist),
            wh: Array2::random((hidden_dim, hidden_dim),dist),
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
        x: &Array2<f32>, //入力データ
        y: &Array2<f32>, //正解データ
        lr: f32,         //学習率
        epochs: usize,   //エポック数
        batch_size: usize, //バッチサイズ
        loss: &Loss,     //損失関数
    ) {
        let num_samples=x.nrows();
        let mut indices: Vec<usize>=(0..num_samples).collect();
        let mut rng=thread_rng();

        for epoch in 0..epochs{
            indices.shuffle(&mut rng);

            for chunk in indices.chunks(batch_size){
                let mut batch_grads=RnnGrads{
                    d_wx:Array2::zero(self.wx.dim()),
                    d_wh:Array2::zero(self.wh.dim()),
                    d_b:Array1::zero(self.b.dim()),
                };
                for &idx in chunk{
                    let single_x=x.slice(s![idx, .., ..]).to_owned();
                    let single_y=y.slice(s![idx, .., ..]).to_owned();

                    let (output,history)=self.forward_train(&single_x);
                    let y_pred=match output{
                        RnnOutput::Sequences(arr)=>arr,
                        _ => unreachable!(),
                    };

                    //逆伝播
                    let delta=loss.grad(&y_pred,&single_y);
                    let grads=self.backward(&single_x,&history,&delta);

                    //蓄積
                    batch_grads.d_wx+=&grads.d_wx;
                    batch_grads.d_wh+=&grads.d_wh;
                    batch_grads.d_b+=&grads.d_b;
                }
                let avg_lr=lr/chunk.len() as f32;
                self.update(&batch_grads,avg_lr);
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
