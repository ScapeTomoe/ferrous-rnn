pub struct RnnGrads {
    pub d_wx: Array2<f32>,
    pub d_wh: Array2<f32>,
    pub d_b:  Array1<f32>,
}