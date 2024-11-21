use candle_core::*;
use candle_nn::*;
use ops::sigmoid;

/// !https://github.com/mlech26l/ncps/blob/master/ncps/torch/ltc_cell.py
pub struct LtcCell {
    input_lin: Linear,
    hidden_lin: Linear,
    out_lin: Linear,
    activation: Activation,
    tau: Tensor,
    a: Tensor,
    sensory_weight: Linear,
    weights: Linear,
    sensory_mu: Tensor,
}

impl LtcCell {
    pub fn new(input_dim: usize, hidden_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let a = Self {
            activation: Activation::GeluPytorchTanh,
            input_lin: linear(input_dim, hidden_dim, vb.clone())?,
            hidden_lin: linear(hidden_dim, hidden_dim, vb.clone())?,
            out_lin: linear(hidden_dim, out_dim, vb.clone())?,
            tau: Tensor::from_slice(&[1.0f32], 1, vb.clone().device())?,
            a: Tensor::from_slice(&[1.0f32], 1, vb.device())?,
            sensory_weight: linear_no_bias(1, 1, vb.clone())?, // TODO: set shape
            weights: linear_no_bias(1, 1, vb.clone())?, // TODO: set shape
            sensory_mu: Tensor::from_slice(&[1.0f32], 1, vb.device())?, // TODO
        };
        Ok(a)
    }

    fn map_input(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.input_lin.forward(input)?;
        Ok(x)
    }

    fn map_outpu(&self, state: &Tensor) -> Result<Tensor> {
        let x = self.out_lin.forward(state)?;
        Ok(x)
    }

    fn fused_state(&self, state: &Tensor, dt: &Tensor, input: &Tensor) -> Result<Tensor> {
        let one = Tensor::ones(1, state.dtype(), state.device())?;
        let tau_inv = (one.clone()/self.tau.clone())?;
        let h = if input.dims()[0] > 0 { 
            Tensor::cat(&[state.clone(), input.clone()], 1)?
        } else {
            state.clone()
        };

        let h = self.hidden_lin.forward(&h)?.broadcast_add(&self.input_lin.forward(&h)?)?;
        let f_x = Activation::GeluPytorchTanh.forward(&h)?;


        let num_2 = dt.broadcast_mul(&f_x)?.broadcast_mul(&self.a)?;
        let denom = one.broadcast_add(&dt.broadcast_mul(&tau_inv.broadcast_add(&f_x)?)?)?;

        let x = input.broadcast_div(&denom)?.broadcast_add(&num_2.broadcast_div(&denom)?)?;
        Ok(x)
    }
    
    pub fn solve_ode(&self, x: &Tensor, inputs: &Tensor, elapsed_time: f32) -> Result<Tensor> {
        let v_pre = x.clone();
        const ODE_UNFOLD: usize = 10;

        // Pre-compute the effects of the sensory neurons 
        let sensory_w_activation = self.sensory_weight.forward(&sigmoid(inputs)?)?; // TODO: Sigmoid with sensory_mu
        let cm_t = elapsed_time / ODE_UNFOLD as f32;

        
        for t in 0..ODE_UNFOLD {
            let w_activation = self.weights.forward(&sigmoid(&v_pre)?)?; // TODO: Sigmoid with sensory_mu
            
        
        }   
       
        Ok(v_pre)
    }
}

impl LtcCell {
    /// Returns the h and next state
    fn forward(&self, inputs: &Tensor, states: &Tensor, elapsed_time: f32) -> Result<(Tensor, Tensor)> {
        let inputs = self.map_input(inputs)?;
        
        let states = self.solve_ode(states, &inputs, elapsed_time)?;
        
        let outputs = self.map_outpu(&states)?;
        Ok((outputs, states))
    }
}

pub struct LtcNet {
    ltc_layers: Vec<LtcCell>,
}

//impl LtcNet {
//    pub fn new(state_dim: usize, input_dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
//        let mut ltc_layers = Vec::new();
//        for _ in 0..num_layers {
//            ltc_layers.push(LtcCell::new(state_dim, 32,input_dim, vb.clone())?);
//        }
//        Ok(Self { ltc_layers })
//   // }

//    pub fn forward(&self, x: &Tensor, i: &Tensor) -> Result<(Tensor, Tensor)> {
//        let mut x = x.clone();
//        let mut i = i.clone();
//        for ltc_layer in &self.ltc_layers {
//            i = ltc_layer.forward(&x, &i)?;
//        }
//        Ok((x, i))
//    }
//    
//}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltc_layer() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let ltc = LtcCell::new(2, 32, 2, vb)?;
        let x = Tensor::from_slice(&[1.0f32, 1.0f32], &[1,2], &device.clone())?;
        let input = Tensor::zeros((32), DType::F32, &device)?;
        Ok(())
    }

    #[test]
    fn test_ltc_neural_network() -> Result<()> {
      //  let device = Device::cuda_if_available(0)?;
      //  let varmap = VarMap::new();
      //  let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
//
      //  let ltc_net = LtcNet::new(1, 1, 2, vb)?;
      //  let x = Tensor::from_slice(&[1.0f32, 1.0f32], &[1,2], &device.clone())?;
      //  let y = ltc_net.forward(&x, &x)?;
        Ok(())
    }
}