use rand::prelude::*;

//features: [0]: Square Feet, [1]: Bedrooms, [2]: Age
pub fn generate_training_dataset(amount: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = rand::rng();

    let mut training: Vec<Vec<f64>> = Vec::new();
    let mut actual_value: Vec<Vec<f64>> = Vec::new();

    for _ in 0..amount {
        let square_feet = rng.random_range(800.0..=4000.0);
        let bedrooms = rng.random_range(1.0..=7.0);
        let age = rng.random_range(0.0..=60.0);

        let price = 100000.0 + square_feet * 180.0 + bedrooms * 25000.0 - age * 1500.0
            + rng.random_range(-20000.0..=20000.0);

        training.push(vec![square_feet, bedrooms, age]);
        actual_value.push(vec![scale_price(&price, 219000.0, 935000.0)]);
    }
    (training, actual_value)
}

pub fn scale_price(price: &f64, min_price: f64, max_price: f64) -> f64 {
    2.0 * (price - min_price) / (max_price - min_price) - 1.0
}

pub fn unscale_price(scaled_price: &f64, min_price: f64, max_price: f64) -> f64 {
    (scaled_price + 1.0) / 2.0 * (max_price - min_price) + min_price
}

//standardize the data
pub fn standardize_data(dataset: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let training_size: f64 = dataset.len() as f64;
    let mut res: Vec<Vec<f64>> = dataset.clone();

    for j in 0..dataset[0].len() {
        let mean = dataset.iter().map(|row| row[j]).sum::<f64>() / training_size;
        let standard_deviation =
            (dataset.iter().map(|row| (row[j] - mean).powi(2)).sum::<f64>() / training_size).sqrt();
        //z-score
        for i in 0..dataset.len() {
            res[i][j] = (dataset[i][j] - mean) / standard_deviation; // z-score, column j only
        }
    }
    res
}
