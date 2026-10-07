use rand::prelude::*;

//features: [0]: Square Feet, [1]: Bedrooms, [2]: Age
pub fn generate_training_dataset(amount: usize) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut rng = rand::rng();

    let mut training: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut actual_value: Vec<Vec<f64>> = Vec::new();

    for _ in 0..amount {
        let square_feet = rng.random_range(800.0..=4000.0);
        let bedrooms = rng.random_range(1.0..=7.0);
        let age = rng.random_range(0.0..=60.0);

        let price = 100000.0 + square_feet * 180.0 + bedrooms * 25000.0 - age * 1500.0
            + rng.random_range(-20000.0..=20000.0);

        training.push(vec![vec![square_feet], vec![bedrooms], vec![age]]);
        actual_value.push(vec![price]);
    }
    (training, actual_value)
}
