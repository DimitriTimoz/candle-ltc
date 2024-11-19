use candle_core::*;
use candle_nn::*;

pub struct Ltc {
    pub dense_layer: Linear,
    activation: Activation,
    tau: Tensor,
    a: Tensor,
}

impl Ltc {
    pub fn new(state_dim: usize, input_dim: usize, vb: VarBuilder) -> Result<Self> {
        let a = Ltc {
            activation: Activation::GeluPytorchTanh,
            dense_layer: linear(state_dim + input_dim, state_dim, vb.clone())?,
            tau: Tensor::from_slice(&[1.0f32], 1, vb.clone().device())?,
            a: Tensor::from_slice(&[1.0f32], 1, vb.device())?,
        };
        Ok(a)
    }

    fn fused_state(&self, x: &Tensor, dt: &Tensor, i: &Tensor) -> Result<Tensor> {
        let one = Tensor::ones(1, x.dtype(), x.device())?;
        let tau_inv = (one.clone()/self.tau.clone())?;
        let h = if i.dims()[0] > 0 { 
            Tensor::cat(&[x.clone(), i.clone()], 1)?
        } else {
            x.clone()
        };

        let f = self.dense_layer.forward(&h)?;
        let f = self.activation.forward(&f)?;

        let denom = (f.clone().broadcast_add(&tau_inv)?).broadcast_mul(&(one + dt.clone())?)?;

        let x = ((x.broadcast_div(&denom))?.broadcast_add(&((f.broadcast_mul(dt)?.broadcast_mul(&self.a)?)/(denom.clone()))?))?;
        Ok(x)
    }
    
    pub fn solve_ode(&self, x: &Tensor, dt: Tensor, i: Tensor) -> Result<Tensor> {
        const RANGE: u8 = 10;

        let mut dt = dt;
        let mut x = x.clone();
        let sub_dt = (dt.clone()/Tensor::from_slice(&[RANGE as f32], &[1], x.device()))?;

        for _ in 0..10 {
            x = self.fused_state(&x, &dt, &i)?;
            dt =  (dt + sub_dt.clone())?;
        }
        Ok(x)
    }
}

impl Module for Ltc {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs;
        let dt = Tensor::from_slice(&[0.1f32], 1, x.device())?;
        let i = Tensor::zeros(0, DType::F32, x.device())?;
        let x = self.solve_ode(x, dt, i)?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltc() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let ltc = Ltc::new(2, 1, vb)?;
        let x = Tensor::from_slice(&[1.0f32, 1.0f32], &[1,2], &device.clone())?;
        let y = ltc.forward(&x)?;
        Ok(())
    }
}