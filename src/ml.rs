use crate::Layer;
use crate::numpy::functions as function;
use crate::numpy::matrix_arithmetic as array;

pub fn init_input_layer(training_data: &Vec<Vec<f64>>) -> Layer {
    Layer::new(training_data.clone(), training_data[0].len())
}
pub fn train_model(
    training_data: Vec<Vec<Vec<f64>>>,
    mut model: Vec<Layer>,
    number_of_neurons_hidden_layer: &Vec<usize>,
    number_of_hidden_layers: &usize,
) -> f64 {
    //weights and biases will be updated in this loop
    for i in 0..training_data.len() {}
    0.0
}

pub fn init_model(
    number_of_neurons_hidden_layer: &Vec<usize>,
    input_layer: Layer,
    number_of_hidden_layers: &usize,
) -> Vec<Layer> {
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

pub fn init_hidden_layer(
    number_of_neurons_hidden_layer: &Vec<usize>,
    input_layer: Layer,
    number_of_hidden_layers: &usize,
) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();

    layers.push(input_layer);

    for i in 0..*number_of_hidden_layers {
        let hidden_layer: Layer = Layer::new(
            apply_activation_function(layers[layers.len() - 1].weighted_sum_matrix.clone()),
            number_of_neurons_hidden_layer[i],
        );

        layers.push(hidden_layer);
    }
    layers
}
pub fn weighted_sum_matrix(
    input_matrix: &Vec<Vec<f64>>,
    weight_matrix: &Vec<Vec<f64>>,
    bias_matrix: &Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    array::matrix_addition(
        &array::matrix_multiplication(weight_matrix, input_matrix),
        bias_matrix,
    )
}

pub fn apply_activation_function(weighted_sum_matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut matrix: Vec<Vec<f64>> =
        vec![vec![0.0; weighted_sum_matrix[0].len()]; weighted_sum_matrix.len()];
    //dimension of weighted sum matrix is always nx1
    for i in 0..weighted_sum_matrix.len() {
        matrix[i][0] = function::sigmoid(weighted_sum_matrix[i][0]);
    }
    matrix
}
