use std::f64::consts::E as e;
mod numpy;
use numpy as np;

fn main() {
    let number_of_neurons = 10;
    let mut input_matrix: Vec<Vec<f64>> = np::create_matrix(number_of_neurons - 1, 1);
    //input_matrix.len() gives the number of rows
    //input_matrix.len() + 1 == number_of_neurons
    let mut weight_matrix: Vec<Vec<f64>> = np::create_matrix(input_matrix.len() + 1, input_matrix.len());
    let mut bias_matrix: Vec<Vec<f64>> = np::create_matrix(input_matrix.len() + 1, 1);
    let mut weighted_sum_matrix = weighted_sum_matrix(input_matrix, weight_matrix, bias_matrix);
    dbg!(weighted_sum_matrix);
}

fn weighted_sum_matrix(input_matrix: Vec<Vec<f64>>, weight_matrix: Vec<Vec<f64>>, bias_matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    np::matrix_addition(np::matrix_multiplication(weight_matrix, input_matrix), bias_matrix)
}

fn sigmoid(z: f64) -> f64{
    let base: f64 = e;
    1.0/(1.0+base.powf(-z))
}