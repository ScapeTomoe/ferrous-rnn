fn relu(x: f32) -> f32 {

    #[cfg(target_arch = "x86_64")]
    {
        let result: f32;
        unsafe {
            std::arch::asm!(
                "xorps {zero},{zero}",
                "maxss {zero},{input}",
                input = in(xmm_reg) x,
                zero = out(xmm_reg) result,
            );
        }
        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        if x > 0.0 { x } else { 0.0 }
    }
}

fn sigmoid(x:f32) ->f32{
    1.0/(1.0+f32::exp(-x))
}

fn tanh(x:f32) ->f32{
    x.tanh()
}

fn leaky_relu(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        let alpha: f32 = 0.01;
        let result: f32;
        unsafe {
            std::arch::asm!(
                "xorps {zero},{zero}",
                "maxss {zero},{input}",
                "mulss {alpha},{neg}",
                "minss {neg},{input2}",
                "addss {zero},{neg}",
                input  = in(xmm_reg) x,
                input2 = in(xmm_reg) x,
                alpha  = in(xmm_reg) alpha,
                zero   = out(xmm_reg) result,
                neg    = out(xmm_reg) _,
            );
        }
        result
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if x > 0.0 { x } else { 0.01 * x }
    }
}

fn elu(x:f32) ->f32{
    if x>0.0{
        x
    } else{
        x.exp()-1.0
    }
}

fn relu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn tanh_grad(x: f32) -> f32 {
    1.0 - x.tanh().powi(2)
}

fn leaky_relu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.01 }
}

fn elu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { x.exp() }
}

pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu,
    Elu,
}

impl Activation {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::Relu      => relu(x),
            Activation::Sigmoid   => sigmoid(x),
            Activation::Tanh      => tanh(x),
            Activation::LeakyRelu => leaky_relu(x),
            Activation::Elu       => elu(x),
        }
    }

    pub fn grad(&self, x: f32) -> f32 {
        match self {
            Activation::Relu      => relu_grad(x),
            Activation::Sigmoid   => sigmoid_grad(x),
            Activation::Tanh      => tanh_grad(x),
            Activation::LeakyRelu => leaky_relu_grad(x),
            Activation::Elu       => elu_grad(x),
        }
    }
}