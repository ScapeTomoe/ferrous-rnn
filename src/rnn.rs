pub struct Rnn{
    wx:Array2<f32>,
    wh:Array2<f32>,
    b:Array1<f32>,
    hidden_dim: usize,
    activation: Activation
}

impl Rnn{
    pub fn new(input_dim:usize,hidden_dim:usize,activation:Activation) ->Self{
        Rnn{
            wx:Array2::zeros((hidden_dim,input_dim)),
            wh:Array2::zeros((hidden_dim,hidden_dim)),
            b: Array1::zeros(hidden_dim),
            hidden_dim,
            activation,
        }
    }
}

pub enum RnnOutput {
    Sequences(Array2<f32>),  // (seq_len, hidden_dim)
    LastOnly(Array1<f32>),   // (hidden_dim,)
}

pub enum OutputMode {
    Sequences,
    LastOnly,
}

// 推論用（速い、メモリ少ない）
pub fn forward(&self, x:&Array2<f32>,mode:OutPutMode) ->RnnOutput{
    let mut h=Array1::zeros(self.hidden_dim);
    let mut history=vec![];

    for t in 0..x.nrows(){
        let x_t=x.row(t);
        let z=self.wx.dot(&x_t)+self.wh.dot(&h)+&self.b;
        h=z.mapv(|v| self.activation.apply(v));
        history.push(h.clone());
    }

    match mode{
        OutputMode::Sequences =>{
            let views: Vec<_> =history.iter().map(|h| h.view().insert_axis(Axis(0))).collect();
            RnnOutput::Sequences(ndarray::stack(Axis(0),&views).unwrap())
        },
        OutputMode::LastOnly =>RnnOutput::LastOnly(h),
    }
}
// 学習用（historyも返す）
pub fn forward_train(&self, x: &Array2<f32>) -> (RnnOutput, Vec<Array1<f32>>){
    let mut h = Array1::zeros(self.hidden_dim);
    let mut history = vec![];

    for t in 0..x.nrows() {
        let x_t = x.row(t);
        let z = self.wx.dot(&x_t) + self.wh.dot(&h) + &self.b;
        h = z.mapv(|v| self.activation.apply(v));
        history.push(h.clone());
    }

    let views: Vec<_> = history.iter().map(|h| h.view().insert_axis(Axis(0))).collect();
    let sequences = RnnOutput::Sequences(stack(Axis(0), &views).unwrap());

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
    let mut d_b  = Array1::zeros(self.b.dim());
    let mut d_h  = Array1::zeros(self.hidden_dim);

    for t in (0..seq_len).rev() {
        let h_prev = if t == 0 { Array1::zeros(self.hidden_dim) } else { history[t-1].clone() };

        // δt・f'(zt)
        let d_raw = (&delta.row(t) + &d_h) * history[t].mapv(|v| self.activation.grad(v));

        // 勾配の累積
        d_wx = d_wx + d_raw.view().insert_axis(Axis(1)).dot(&x.row(t).insert_axis(Axis(0)));
        d_wh = d_wh + d_raw.view().insert_axis(Axis(1)).dot(&h_prev.view().insert_axis(Axis(0)));
        d_b  = d_b + &d_raw;

        // 前ステップへの誤差伝播
        d_h = self.wh.t().dot(&d_raw);
    }

    RnnGrads { d_wx, d_wh, d_b }
}