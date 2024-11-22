use std::ops::{Index, RangeBounds};

use candle_core::*;

#[derive(Clone, Copy)]
pub enum Polarity {
    Excitatory,
    Inhibitory,
}

pub struct Wiring {
    adjacency_matrix: Tensor,
    input_dim: Option<usize>,
    output_dim: Option<usize>,
    sensory_adjacency_matrix: Option<Tensor>,
}

impl Wiring {
    pub fn new(units: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            adjacency_matrix: Tensor::zeros((units, units), DType::I64, device)?,
            input_dim: None,
            output_dim: None,
            sensory_adjacency_matrix: None,
        })
    }

    pub fn add_sinapse(&mut self, from: usize, to: usize, polarity: Polarity) -> Result<()> {
        // Check if the sinapse is valid
        if from >= self.adjacency_matrix.dim(0)? || to >= self.adjacency_matrix.dim(1)? {
            bail!("Invalid sinapse from {} to {}", from, to);
        }

        let value = match polarity {
            Polarity::Excitatory => 1i64,
            Polarity::Inhibitory => -1i64,
        };

        self.adjacency_matrix.set(&[from, to], value)?;
        Ok(())
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
    fn test_wiring() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let mut wiring = Wiring::new(32, &device)?;
        wiring.add_sinapse(0, 1, Polarity::Excitatory)?;
        Ok(())
    }
}