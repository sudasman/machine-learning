pub mod matrix_arithmetic {
    use rand::prelude::*;

    //creates a nxm matrix with random entries
    pub fn create_matrix(n: usize, m: usize, weight_matrix_or_bias_matrix: bool) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        if !weight_matrix_or_bias_matrix {
            return (0..n)
                .map(|_| (0..m).map(|_| rng.random_range(-100.0..=100.0)).collect())
                .collect();
        }
        (0..n)
            .map(|_| (0..m).map(|_| rng.random_range(-1.0..=1.0)).collect())
            .collect()
    }

    pub fn create_matrix_empty(n: usize, m: usize) -> Vec<Vec<f64>> {
        vec![vec![0.0; m]; n]
    }

    pub fn matrix_multiplication(
        matrix1: &Vec<Vec<f64>>,
        matrix2: &Vec<Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        //linear algebra 101
        assert_eq!(matrix1[0].len(), matrix2.len());

        let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; matrix2[0].len()]; matrix1.len()];
        for i in 0..matrix1.len() {
            for j in 0..matrix2[0].len() {
                for k in 0..matrix2.len() {
                    matrix[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        matrix
    }

    pub fn matrix_addition(matrix1: &Vec<Vec<f64>>, matrix2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        assert_eq!(
            (matrix1.len(), matrix1[0].len()),
            (matrix2.len(), matrix2[0].len())
        );

        //either matrix1 or matrix2 works here
        let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; matrix1[0].len()]; matrix1.len()];
        for i in 0..matrix1.len() {
            for j in 0..matrix1[0].len() {
                matrix[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        matrix
    }
}

pub mod functions {
    //for scope reasons
    use crate::housing_price_dataset::scale_price;
    use std::f64::consts::E as e;
    //Use Case: Binary Classification (Yes/No)
    pub fn sigmoid(z: f64) -> f64 {
        let base: f64 = e;
        1.0 / (1.0 + base.powf(-z))
    }

    //Use Case: Multi Class Classification (Number Recognition)
    pub fn softmax(weighted_sum_matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; weighted_sum_matrix[0].len()]; weighted_sum_matrix.len()];

        let mut denominator_sum: f64 = 0.0;

        for i in 0..weighted_sum_matrix.len() {
            denominator_sum += weighted_sum_matrix[i][0].exp()
        }

        for i in 0..weighted_sum_matrix.len() {
            matrix[i][0] = weighted_sum_matrix[i][0].exp() / denominator_sum;
        }
        matrix
    }

    pub fn MSE(prediction: &Vec<Vec<f64>>, actual_value: &Vec<Vec<f64>>) -> f64 {
        let mut sum: f64 = 0.0;

        for i in 0..prediction.len() {
            let scale_actual_value: f64 = scale_price(&actual_value[i][0], 219000.0, 935000.0);
            let error: f64 = scale_actual_value - prediction[i][0];
            let squared_error = error.powf(2.0);
            sum += squared_error;
        }

        //219000.0, 935000.0
        (1.0 / prediction.len() as f64) * sum
    }
}
