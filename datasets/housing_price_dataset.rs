use rand::prelude::*;

//features: [0]: Square Feet, [1]: Bedrooms, [2]: Age
pub fn generate_training_dataset(amount: usize) -> (Vec<Vec<f64>>>, Vec<Vec<f64>>) {
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
        actual_value.push(vec![price]);
    }
    (training, actual_value)
}

//standardize the data 
pub fn standardize_data(training_dataset: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let training_size: f64 = training_dataset.len() as f64;
    let mut res = training_size.clone();
    
    for i in 0..training_dataset[0].len() {
        let mean = training_dataset.iter().map(|row|).sum::<f64>() / training_size;
        let standard_deviation = (x.iter().map(|row| (row[i] - mean).powi(2)).sum::<f64>() / training_size).sqrt();
        //z-score
        res.iter().zip(training_dataset).map(|(row1, row2)| row1.iter().zip(row2).map(|(val1, val2)| (val2 - mean) / standard_deviation).collect()).collect();
    }
    res
}
