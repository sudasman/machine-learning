mod numpy;
#[path = "../datasets/housing_price_dataset.rs"]
mod housing_price_dataset;
use crate::housing_price_dataset::{generate_training_dataset as generate_housing, unscale_price as price};
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
        let weighted_sum_matrix = weighted_sum_matrix(&input_matrix, &weight_matrix, &bias_matrix);

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
    let mut layers: Vec<Layer> = vec![];
    let data: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) = generate_housing(5);
    let input_layer_neurons: usize = 10;
    //array::create_matrix(input_layer_neurons - 1, 1, false)
    let input_layer: Layer = Layer::new(
        data.0[0].clone(),
        input_layer_neurons,
    );

    let number_of_neurons_hidden_layer = vec![8];
    let number_of_hidden_layers = 1;
    layers = init_hidden_layer(number_of_neurons_hidden_layer, input_layer, number_of_hidden_layers);

    let output_layer: Layer = Layer::new(
        apply_activation_function(layers[layers.len() - 1].weighted_sum_matrix.clone()),
        1,
    );
    layers.push(output_layer);

    dbg!(layers[2].weighted_sum_matrix.clone());
    dbg!(function::MSE(&layers[2].weighted_sum_matrix.clone(), &data.1));
}

fn init_hidden_layer(neurons: Vec<usize>, input_layer: Layer, number_of_hidden_layers: usize) -> Vec<Layer>{
    let mut layers: Vec<Layer> = Vec::new();

    layers.push(input_layer);

    for i in 0..number_of_hidden_layers{
        let hidden_layer: Layer = Layer::new(
            apply_activation_function(layers[layers.len() - 1].weighted_sum_matrix.clone()),
            neurons[i],
        );

        layers.push(hidden_layer);
    }
    layers
}
fn weighted_sum_matrix(
    input_matrix: &Vec<Vec<f64>>,
    weight_matrix: &Vec<Vec<f64>>,
    bias_matrix: &Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    array::matrix_addition(
        &array::matrix_multiplication(weight_matrix, input_matrix),
        bias_matrix,
    )
}

fn apply_activation_function(weighted_sum_matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut matrix: Vec<Vec<f64>> =
        vec![vec![0.0; weighted_sum_matrix[0].len()]; weighted_sum_matrix.len()];
    //dimension of weighted sum matrix is always nx1
    for i in 0..weighted_sum_matrix.len() {
        matrix[i][0] = function::sigmoid(weighted_sum_matrix[i][0]);
    }
    matrix
}
