
pub use burn::{backend::wgpu::WgpuDevice, data::dataset::Dataset, prelude::*};
pub use serde::{Deserialize, Serialize};

pub type Backend = burn::backend::Wgpu;
pub type AutoDiffBackend = burn::backend::Autodiff<Backend>;

fn nullable<'de, D, T, E>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    Option<T>: serde::Deserialize<'de>,
    T: std::str::FromStr<Err = E>,
    E: std::error::Error,
{
    let val = String::deserialize(deserializer)?;
    if val.is_empty() || val == "NA" {
        Ok(None)
    } else {
        val.parse().map(Some).map_err(serde::de::Error::custom)
    }
}

fn get_dataset() -> burn::data::dataset::InMemDataset<HouseTiny> {
    let mut rdr = csv::ReaderBuilder::new();
    let rdr = rdr.has_headers(true);
    burn::data::dataset::InMemDataset::from_csv("data/house_tiny.csv", rdr).unwrap()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HouseTiny {
    #[serde(rename = "NumRooms")]
    #[serde(deserialize_with = "nullable")]
    num_rooms: Option<usize>,
    #[serde(rename = "RoofType")]
    #[serde(deserialize_with = "nullable")]
    roof_type: Option<String>,
    #[serde(rename = "Price")]
    price: usize,
}

#[cfg(test)]
mod chapter_2 {
    use super::*;

    #[test]
    fn two_one_one() {
        let device = Default::default();
        let x = Tensor::<Backend, 1, Int>::arange(0..12, &device);
        println!("{:?}", x.to_data());

        let x = x.reshape([3, 4]);
        println!("{:?}", x.to_data());

        println!(
            "{:?}",
            Tensor::<Backend, 3, Int>::zeros([2, 3, 4], &device).to_data()
        );

        println!(
            "{:?}",
            Tensor::<Backend, 3, Int>::ones([2, 3, 4], &device).to_data()
        );

        println!(
            "{:?}",
            Tensor::<Backend, 3>::random(
                [2, 3, 4],
                burn::tensor::Distribution::Uniform(-1., 1.),
                &device
            )
            .to_data(),
        );

        println!(
            "{:?}",
            Tensor::<Backend, 3>::from_floats(
                [[[0., 1., 2.], [3., 4., 5.]], [[6., 7., 8.], [9., 10., 11.]]],
                &device
            )
            .to_data()
        );
    }

    #[test]
    fn two_one_two() {
        let device = Default::default();
        let x = Tensor::<Backend, 1, Int>::arange(0..12, &device);
        let x = x.reshape([3, 4]);
        println!("{:?}", x.clone().slice([2..3]).to_data());
        println!("{:?}", x.clone().slice([1..3]).to_data());
        let x = x.slice_assign(
            [1..2, 2..3],
            Tensor::<Backend, 2, Int>::from_data([[17]], &device),
        );
        println!("{:?}", x.to_data());
        let x = x.slice_assign(
            [0..2, 0..4],
            Tensor::<Backend, 2, Int>::full([2, 4], 12, &device),
        );
        println!("{:?}", x.to_data());
    }

    #[test]
    fn two_one_three() {
        let device = Default::default();
        let x = Tensor::<Backend, 2>::random(
            [3, 4],
            burn::tensor::Distribution::Uniform(-1., 1.),
            &device,
        );
        println!("{:?}", x.to_data());

        let x = Tensor::<Backend, 1>::from_data([1.0, 2., 4., 8.], &device);
        let y = Tensor::<Backend, 1>::from_data([2., 2., 2., 2.], &device);
        println!("{:?}", x.clone().add(y.clone()).to_data());
        println!("{:?}", x.clone().sub(y.clone()).to_data());
        println!("{:?}", x.clone().mul(y.clone()).to_data());
        println!("{:?}", x.clone().div(y.clone()).to_data());
        println!("{:?}", x.clone().powf(y.clone()).to_data());

        let x = Tensor::<Backend, 1, Int>::arange(0..12, &device)
            .reshape([3, 4])
            .float();
        let y = Tensor::<Backend, 2>::from_data(
            [[2., 1., 4., 3.], [1., 2., 3., 4.], [4., 3., 2., 1.]],
            &device,
        );
        println!(
            "{:?}",
            Tensor::<Backend, 2>::cat(vec![x.clone(), y.clone()], 0).to_data()
        );
        println!(
            "{:?}",
            Tensor::<Backend, 2>::cat(vec![x.clone(), y.clone()], 1).to_data()
        );
        println!("{:?}", x.clone().equal(y.clone()).to_data());
        println!("{:?}", x.sum().to_data());
    }

    #[test]
    fn two_one_four() {
        let device = Default::default();
        let x = Tensor::<Backend, 1, Int>::arange(0..3, &device).reshape([3, 1]);
        let y = Tensor::<Backend, 1, Int>::arange(0..2, &device).reshape([1, 2]);
        println!("{:?}", x.add(y).to_data());
    }
}

#[cfg(test)]
mod chapter_2_2 {
    use super::*;

    #[test]
    fn two_two_one() {
        // let device = Default::default();
        let dataset = get_dataset();
        println!("{:?}", dataset.len());
    }

    #[test]
    fn two_two_three() {
        let device = Default::default();
        let dataset = get_dataset();
        let data = dataset.iter().map(|item| {
            Data::<f32, 1>::from([
                item.num_rooms.unwrap_or(3) as f32,
                if item.roof_type == Some("Slate".to_string()) {
                    1.
                } else {
                    0.
                },
                item.price as f32,
            ])
        });
        let tensors: Vec<_> = data
            .map(|data| Tensor::<Backend, 1>::from_data(data, &device).reshape([1, -1]))
            .collect();
        let x = Tensor::<Backend, 2>::cat(tensors, 0);
        println!("{:?}", x.to_data());
    }
}

#[cfg(test)]
mod chapter_2_3 {
    use super::*;

    type SimpleTensor<const D: usize> = Tensor<Backend, D, Float>;

    fn simple_arange(range: std::ops::Range<i64>, device: &WgpuDevice) -> SimpleTensor<1> {
        Tensor::<Backend, 1, Int>::arange(range, device).float()
    }

    #[test]
    fn two_three_one() {
        let device = Default::default();
        let x = SimpleTensor::<1>::from_data([3.], &device);
        let y = SimpleTensor::<1>::from_data([2.], &device);
        println!("{:?}", x.clone().add(y.clone()).to_data());
        println!("{:?}", x.clone().mul(y.clone()).to_data());
        println!("{:?}", x.clone().div(y.clone()).to_data());
        println!("{:?}", x.clone().powf(y.clone()).to_data());
    }

    #[test]
    fn two_three_five() {
        let device = Default::default();
        let a = simple_arange(0..6, &device).reshape([2, 3]);
        let b = a.clone();
        println!("{:?}", a.clone().add(b.clone()).to_data());
        println!("{:?}", a.clone().mul(b.clone()).to_data());
        let a_scale = 2.;
        let X = simple_arange(0..24, &device).reshape([2, 3, 4]);
        println!("{:?}", X.clone().add_scalar(a_scale).to_data());
    }

    #[test]
    fn two_three_six() {
        // Same as two_three_seven, basically.
        let device = Default::default();
        let a = simple_arange(0..3, &device);
        println!("{:?}", a.sum().to_data());
        let a = simple_arange(0..6, &device).reshape([2, 3]);
        // Sum_dim only available with keep_dim = True
        println!("{:?}", a.clone().sum_dim(0).to_data());
        println!("{:?}", a.clone().sum_dim(1).to_data());
        println!(
            "{:?}",
            a.clone()
                .sum()
                .div_scalar(a.shape().num_elements() as f32)
                .to_data()
        );
        println!("{:?}", a.clone().mean_dim(0).to_data());
    }

    #[test]
    fn two_three_eight() {
        let device = Default::default();
        let x = simple_arange(0..3, &device);
        let y = SimpleTensor::<1>::ones([3], &device);
        // No .dot method available
        println!("{:?}", x.clone().mul(y.clone()).sum().to_data());
    }

    #[test]
    fn two_three_nine() {
        let device = Default::default();
        let a = simple_arange(0..6, &device).reshape([2, 3]);
        let x = simple_arange(0..3, &device).reshape([3, 1]);
        // No vector multiply?
        println!("{:?}", a.clone().matmul(x.clone()).to_data());
        // println!("{:?}", a.clone().mul(x.clone()).to_data());
    }

    #[test]
    fn two_three_ten() {
        let device = Default::default();
        let a = simple_arange(0..6, &device).reshape([2, 3]);
        let b = SimpleTensor::<2>::ones([3, 4], &device);
        println!("{:?}", a.clone().matmul(b.clone()).to_data());
    }

    #[test]
    fn two_three_eleven() {
        let device = Default::default();
        let u = SimpleTensor::<1>::from_data([3., -4.], &device);
        // No .norm method available
        println!("{:?}", u.clone().powf_scalar(2.).sum().sqrt().to_data());
        println!("{:?}", u.clone().abs().sum().to_data());
        let f = SimpleTensor::<2>::ones([4, 9], &device);
        println!("{:?}", f.clone().powf_scalar(2.).sum().sqrt().to_data());
    }
}

mod chapter_2_4 {
    use super::*;

    #[test]
    fn two_four_one() {
        let f = |x: f32| {
            3. * x.powf(2.) - 4. * x
        };
        for ih in 1..6 {
            let h = 10f32.powf(-ih as f32);
            println!("h={:.5}, numerical limit={}", h, (f(1. + h) - f(1.)) / h);
        }
    }

    #[test]
    fn two_four_two() {
        // No visualization available
    }

    #[test]
    fn two_four_three() {
        // No code.
    }

    #[test]
    fn two_four_four() {
        // No code.
    }
}

#[cfg(test)]
mod chapter_2_5 {
    use super::*;

    #[test]
    fn two_five_one() {
        let device = Default::default();
        let x = Tensor::<AutoDiffBackend, 1, Int>::arange(0..4, &device)
            .float()
            .require_grad();
        let y = x.clone().mul(x.clone()).sum().mul_scalar(2.);
        let grad = y.backward();
        println!("{:?}", x.grad(&grad).map(|t| t.to_data()));

        // Gradients are handled a bit differently in burn.
        // The gradient is stored in a separate bucket, so we don't need to zero it out.
        let y = x.clone().sum();
        let new_grad = y.backward();
        println!("{:?}", x.grad(&new_grad).map(|t| t.to_data()));
    }
}