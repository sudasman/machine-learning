#[path = "../datasets/housing_price_dataset.rs"]
mod housing_price_dataset;
mod ml;
mod numpy;
use crate::housing_price_dataset::{
    generate_training_dataset as generate_housing, unscale_price as price,
};
use ml as pytorch;
use numpy::functions as function;
use numpy::matrix_arithmetic as array;

#[derive(Debug)]
struct Layer {
    neurons: usize,
    input_matrix: Vec<Vec<f64>>,
    weight_matrix: Vec<Vec<f64>>,
    bias_matrix: Vec<Vec<f64>>,
    weighted_sum_matrix: Vec<Vec<f64>>,
}

impl Layer {
    fn new(input_matrix: Vec<Vec<f64>>, neurons: usize) -> Self {
        let weight_matrix: Vec<Vec<f64>> = array::create_matrix(neurons, input_matrix.len(), true);
        let bias_matrix: Vec<Vec<f64>> = array::create_matrix(neurons, 1, true);
        let weighted_sum_matrix = pytorch::weighted_sum_matrix(&input_matrix, &weight_matrix, &bias_matrix);

        Layer {
            neurons,
            input_matrix,
            weight_matrix,
            bias_matrix,
            weighted_sum_matrix,
        }
    }
}

fn main() {
    let number_of_datasets = 5;
    let data: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) = generate_housing(number_of_datasets);
    let number_of_neurons_hidden_layer = vec![8];
    let number_of_hidden_layers = 1;
    let input_layer: Layer = pytorch::init_input_layer(&data.0[0]);
    let model: Vec<Layer> = pytorch::init_model(
        &number_of_neurons_hidden_layer,
        input_layer,
        &number_of_hidden_layers,
    );

    dbg!(pytorch::train_model(
        data.0,
        model,
        &number_of_neurons_hidden_layer,
        &number_of_hidden_layers
    ));
}
