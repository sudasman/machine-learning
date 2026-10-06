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
    fn new(neurons: usize) -> Self{
        let input_matrix: Vec<Vec<f64>> = np::create_matrix(neurons - 1, 1);
        //input_matrix.len() gives the number of rows
        //input_matrix.len() + 1 == number_of_neurons
        let weight_matrix: Vec<Vec<f64>> = np::create_matrix(input_matrix.len() + 1, input_matrix.len());
        let bias_matrix: Vec<Vec<f64>> = np::create_matrix(input_matrix.len() + 1, 1);
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
    let input_layer: Layer = Layer::new(10);

    let mut layers: Vec<Layer> = vec![];

    //create a layer vector that will store all of the layers in this neural network

}

fn weighted_sum_matrix(input_matrix: &Vec<Vec<f64>>, weight_matrix: &Vec<Vec<f64>>, bias_matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    np::matrix_addition(&np::matrix_multiplication(weight_matrix, input_matrix), bias_matrix)
}

fn sigmoid(z: f64) -> f64{
    let base: f64 = e;
    1.0/(1.0+base.powf(-z))
}