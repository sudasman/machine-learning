use std::f64::consts::E as e;
mod numpy;
use numpy as np;

struct Layer{
    neurons: usize,
    input_matrix: Vec<Vec<f64>>,
    weight_matrix: Vec<Vec<f64>>,
    bias_matrix: Vec<Vec<f64>>,
    weighted_sum_matrix: Vec<Vec<f64>>,
}

impl Layer {
    fn new(input_matrix: Vec<Vec<f64>>, neurons: usize) -> Self{
        let weight_matrix: Vec<Vec<f64>> = np::create_matrix(neurons, input_matrix.len());
        let bias_matrix: Vec<Vec<f64>> = np::create_matrix(neurons, 1);
        let weighted_sum_matrix = weighted_sum_matrix(&input_matrix, &weight_matrix, &bias_matrix);
        
        Layer{
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

    let input_layer_neurons: usize = 10;
    let input_layer: Layer = Layer::new(np::create_matrix(input_layer_neurons - 1 , 1),input_layer_neurons);
    layers.push(input_layer);

    let hidden_layer1_neurons: usize = 8;
    let hidden_layer1: Layer = Layer::new(apply_activation_function(layers[layers.len()-1].weighted_sum_matrix.clone()), hidden_layer1_neurons);
    layers.push(hidden_layer1);



}

fn weighted_sum_matrix(input_matrix: &Vec<Vec<f64>>, weight_matrix: &Vec<Vec<f64>>, bias_matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    np::matrix_addition(&np::matrix_multiplication(weight_matrix, input_matrix), bias_matrix)
}

fn apply_activation_function(weighted_sum_matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; weighted_sum_matrix[0].len()]; weighted_sum_matrix.len()];
    //dimension of weighted sum matrix is always nx1
    for i in 0..weighted_sum_matrix.len(){
        matrix[i][0] = sigmoid(weighted_sum_matrix[i][0]);
    }
    matrix
}

fn sigmoid(z: f64) -> f64{
    let base: f64 = e;
    1.0/(1.0+base.powf(-z))
}