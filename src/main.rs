#[path = "../datasets/housing_price_dataset.rs"]
mod housing_price_dataset;
mod numpy;
use crate::housing_price_dataset::{
    generate_training_dataset as generate_housing, unscale_price as price,
};
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
    let mut model: Vec<Layer> = Vec::new();
    let number_of_datasets = 5;
    let data: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) = generate_housing(number_of_datasets);
    let input_layer_neurons: usize = 3;
    let number_of_neurons_hidden_layer = vec![8];
    let number_of_hidden_layers = 1;

    dbg!(train_model(data.0, model, number_of_neurons_hidden_layer, number_of_hidden_layers));
}

fn train_model(training_data: Vec<Vec<Vec<f64>>>, mut model: Vec<Layer>, number_of_neurons_hidden_layer: Vec<usize>, number_of_hidden_layers: usize ) -> f64 {
    for i in 0..training_data.len(){
        let input_layer: Layer = Layer::new(training_data[i].clone(), training_data[i].len());
        
        model = init_model(
            number_of_neurons_hidden_layer.clone(),
            input_layer,
            number_of_hidden_layers,
        );
    }
    0.0
}

fn init_model(number_of_neurons_hidden_layer: Vec<usize>, input_layer: Layer, number_of_hidden_layers: usize, ) -> Vec<Layer> {
    let mut model = init_hidden_layer(
        number_of_neurons_hidden_layer,
        input_layer,
        number_of_hidden_layers,
    );
    let output_layer: Layer = Layer::new(
        apply_activation_function(model[model.len() - 1].weighted_sum_matrix.clone()),
        1,
    );
    model.push(output_layer);
    model
}

fn init_hidden_layer(
    neurons: Vec<usize>,
    input_layer: Layer,
    number_of_hidden_layers: usize,
) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();

    layers.push(input_layer);

    for i in 0..number_of_hidden_layers {
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
