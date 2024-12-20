use candle_core::*;
use rand::seq::SliceRandom;

#[derive(Clone, Copy)]
pub enum Polarity {
    Excitatory = 1,
    Inhibitory = -1,
}

impl Polarity {
    pub fn value(&self) -> i64 {
        *self as i64
    }
}

pub trait WiringTrait {
    fn build(&mut self, input_dim: usize) -> Result<()>;
    fn get_adjacency_matrix(&self) -> &Tensor;
}

pub struct Wiring {
    adjacency_matrix: Tensor,
    input_dim: Option<usize>,
    output_dim: Option<usize>,
    sensory_adjacency_matrix: Option<Tensor>,
    units: usize,
}

impl Wiring {
    pub fn new(units: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            adjacency_matrix: Tensor::zeros((units, units), DType::I64, device)?,
            units,
            input_dim: None,
            output_dim: None,
            sensory_adjacency_matrix: None,
        })
    }

    pub fn set_input_dim(&mut self, input_dim: usize) {
        self.input_dim = Some(input_dim);
        self.sensory_adjacency_matrix = Some(Tensor::zeros((input_dim, self.units), DType::I64, self.adjacency_matrix.device()).unwrap());
    }

    pub fn set_output_dim(&mut self, output_dim: usize) {
        self.output_dim = Some(output_dim);
    }

    pub fn is_built(&self) -> bool {
        self.input_dim.is_some() && self.output_dim.is_some()
    }

    pub fn add_sinapse(&mut self, from: usize, to: usize, polarity: Polarity) -> Result<()> {
        // Check if the sinapse is valid
        if from >= self.adjacency_matrix.dim(0)? || to >= self.adjacency_matrix.dim(1)? {
            bail!("Invalid sinapse from {} to {}", from, to);
        }

        let value = polarity.value();

        self.adjacency_matrix = self.adjacency_matrix.set(&[from, to], value)?;
        Ok(())
    }

    pub fn add_sensory_sinapse(&mut self, from: usize, to: usize, polarity: Polarity) -> Result<()> {
        if !self.is_built() {
            bail!("Cannot add sensory synapses before build() has been called!");
        }

        if let Some(input_dim) = self.input_dim {
            if from >= input_dim {
                bail!("Invalid sensory sinapse from {} to {}", from, to);
            }
        }

        if to >= self.units {
            bail!("Invalid sensory sinapse from {} to {}", from, to);
        }

        let value = polarity.value();
        if let Some(sensory_adjacency_matrix) = self.sensory_adjacency_matrix.as_mut() {
            *sensory_adjacency_matrix = sensory_adjacency_matrix.set(&[from, to], value)?;
        } else {
            unreachable!();
        }

        Ok(())
    }
}

impl WiringTrait for Wiring {
    fn build(&mut self, input_dim: usize) -> Result<()> {
        if let Some(setted_input_dim) = self.input_dim {
            if setted_input_dim != input_dim {
                bail!("Conflicting input dimensions: {} != {}", setted_input_dim, input_dim);
            }
         } else {
             self.set_input_dim(input_dim);
         }
         
         Ok(())
     }

    fn get_adjacency_matrix(&self) -> &Tensor {
        &self.adjacency_matrix
    }
}

pub struct FullyConnected{
    wiring: Wiring,
    self_connection: bool,
}

impl FullyConnected {
    pub fn new(units: usize, self_connection: bool, output_dim: Option<usize>, device: &Device) -> Result<Self> {
        let mut wiring = Wiring::new(units, device)?;
        
        if let Some(output_dim) = output_dim {
            wiring.set_output_dim(output_dim);
        } else {
            wiring.set_output_dim(units);
        }

        for src in 0..units {
            for dst in 0..units {
                if src == dst && !self_connection {
                    continue;
                }

                let polarity = [Polarity::Excitatory, Polarity::Excitatory, Polarity::Inhibitory].choose(&mut rand::thread_rng()).ok_or_else(|| candle_core::Error::Msg("Error choosing polarity".to_owned()))?;
                wiring.add_sinapse(src, dst, *polarity)?;
            }
        }
        Ok(Self {
            wiring,
            self_connection,
        })
    }
}

impl WiringTrait for FullyConnected {
    fn build(&mut self, input_shape: usize) -> Result<()> {
        self.wiring.build(input_shape)?;
        for src in 0..input_shape {
            for dest in 0..self.wiring.units {
                let polarity = [Polarity::Excitatory, Polarity::Excitatory, Polarity::Inhibitory].choose(&mut rand::thread_rng()).ok_or_else(|| candle_core::Error::Msg("Error choosing polarity".to_owned()))?;
                self.wiring.add_sensory_sinapse(src, dest, *polarity)?;
            }
        }

        Ok(())
    }

    fn get_adjacency_matrix(&self) -> &Tensor {
        self.wiring.get_adjacency_matrix()
    }
}


trait HackTraitSet<D: WithDType> {
    fn set(&self, index: &[usize], value: D) -> Result<Tensor>;
}

impl<D: WithDType> HackTraitSet<D> for Tensor {
    fn set(&self, index: &[usize], value: D) -> Result<Self> {
        let n = self.dims().len();
        let shape = vec![1; n];
        let value = Tensor::from_slice(&[value], shape.as_slice(), self.device())?;

        let index = index.iter().map(|&i| i..i+1).collect::<Vec<_>>();
        let t = self.slice_assign(index.as_slice(), &value)?;

        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] 
    fn test_set_trait() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let tensor = Tensor::zeros((2, 2), DType::I64, &device)?;
        let tensor = tensor.set(&[0, 0], 1i64)?;

        assert_eq!(tensor.i((0, 0))?.to_scalar::<i64>()?, 1);
        Ok(())
    }

    #[test]
    fn test_set_sinapses() {
        let device = Device::cuda_if_available(0).unwrap();
        let mut wiring = Wiring::new(4, &device).unwrap();
        
        wiring.add_sinapse(0, 0, Polarity::Excitatory).unwrap();
        wiring.add_sinapse(1, 2, Polarity::Inhibitory).unwrap();

        let adjacency_matrix = wiring.adjacency_matrix.to_vec2::<i64>().unwrap();
        assert_eq!(adjacency_matrix[0][0], 1);
        assert_eq!(adjacency_matrix[1][2], -1);
        assert_eq!(adjacency_matrix[0][1], 0);

        // Test sensory sinapses
        assert!(wiring.add_sensory_sinapse(0, 1, Polarity::Excitatory).is_err());

    }

    #[test]
    fn test_fully_connected() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let mut fc = FullyConnected::new(4, false, Some(2), &device)?;
        fc.build(2)?;
        Ok(())
    }

    #[test]
    fn test_wiring() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let mut wiring = Wiring::new(32, &device)?;
        wiring.add_sinapse(0, 1, Polarity::Excitatory)?;
        Ok(())
    }
}