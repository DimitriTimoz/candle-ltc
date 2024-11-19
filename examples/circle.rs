use candle_core::*;
use candle_nn::*;
use candle_ltc::*;


fn gen_data() -> Result<Tensor> {
    // Generate some sample linear data.
    let ts = Tensor::arange_step(0.0, std::f32::consts::PI, 0.1, &Device::Cpu)?;
    let x = ts.clone().sin()?;
    let y = ts.clone().cos()?;
    let xs = Tensor::stack(&[x.clone(), y.clone()], 0)?;
    Ok(xs)
}

fn main() -> Result<()> {
    let sample_xs = gen_data()?;
    // Use backprop to run a linear regression between samples and get the coefficients back.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = Ltc::new(2, 0, vb)?;
    let params = ParamsAdamW {
        lr: 0.01,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    for step in 0..1000 {
        let mut loss = Tensor::zeros(1, DType::F32, &Device::Cpu)?;
        for i in 0..32 {
            let x = &sample_xs.i((.., i))?.unsqueeze(0)?;
            let ys = model.forward(x)?;
            let mse = (ys.clone().broadcast_sub(x))?.powf(2.0)?.mean(1)?;
            opt.backward_step(&mse)?;
            loss = loss.broadcast_add(&mse)?;
        }
        println!("loss {:?}", loss.to_vec1::<f32>()?); 

    }   
    Ok(())
}
