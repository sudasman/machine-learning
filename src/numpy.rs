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

    pub fn add_bias(matrix: &Vec<Vec<f64>>, bias_matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        assert_eq!(matrix[0].len(), bias_matrix[0].len());
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&bias_matrix[0])
                    .map(|(a, b)| a + b)
                    .collect()
            })
            .collect()
    }

    pub fn matrix_subtraction(matrix1: &Vec<Vec<f64>>, matrix2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        assert_eq!(
            (matrix1.len(), matrix1[0].len()),
            (matrix2.len(), matrix2[0].len())
        );

        let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; matrix1[0].len()]; matrix1.len()];
        for i in 0..matrix1.len() {
            for j in 0..matrix1[0].len() {
                matrix[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }

        matrix
    }
    pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut transposed_matrix: Vec<Vec<f64>> = vec![vec![0.0; matrix.len()]; matrix[0].len()];

        for i in 0..matrix.len() {
            for j in 0..matrix[0].len() {
                transposed_matrix[j][i] = matrix[i][j];
            }
        }
        transposed_matrix
    }

    pub fn sum_columns(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; matrix[0].len()]];

        for i in 0..matrix.len() {
            for j in 0..matrix[i].len() {
                result[0][j] += matrix[i][j];
            }
        }

        result
    }

    pub fn scalar_multiplication(scalar: &f64, matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        matrix
            .iter()
            .map(|i| i.iter().map(|j| scalar * j).collect())
            .collect()
    }

    //the Hadamard product
    pub fn hadamard(matrix1: &Vec<Vec<f64>>, matrix2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        assert_eq!(
            (matrix1.len(), matrix1[0].len()),
            (matrix2.len(), matrix2[0].len())
        );
        //zip returns a tuple
        matrix1
            .iter()
            .zip(matrix2)
            .map(|(matrix1_row, matrix2_row)| {
                matrix1_row
                    .iter()
                    .zip(matrix2_row)
                    .map(|(matrix1_val, matrix2_val)| matrix1_val * matrix2_val)
                    .collect()
            })
            .collect()
    }
}

pub mod functions {
    //for scope reasons
    use super::matrix_arithmetic::*;
    use rand::seq::SliceRandom;
    use rand::*;
    use std::f64::consts::E as e;
    use crate::ml::Layer as Layer;
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
        let sum: f64 = prediction
            .iter()
            .zip(actual_value)
            .map(|(row1, row2)| (row1[0] - row2[0]))
            .sum();
        sum / prediction.len() as f64
    }

    pub fn stochastic_gradient_descent(
        model: &mut Vec<Layer>,
        prediction: &Vec<Vec<f64>>,
        actual: &Vec<Vec<f64>>,
        learning_rate: f64,
    ) {
        let n = prediction.len() as f64;

        let mut delta = scalar_multiplication(&(2.0 / n), &matrix_subtraction(prediction, actual));

        for i in (0..model.len()).rev() {
            let weight_gradient: Vec<Vec<f64>> =
                matrix_multiplication(&transpose(&model[1].input_matrix), &delta);
            let bias_gradient: Vec<Vec<f64>> = sum_columns(&delta);

            let next_delta: Option<Vec<Vec<f64>>> = if 1 > 0 {
                let back = matrix_multiplication(&delta, &transpose(&model[1].weight_matrix));

                //derivative of the sigmoid activation function
                let sigmoid_derivative: Vec<Vec<f64>> = model[1]
                    .input_matrix
                    .iter()
                    .map(|row| row.iter().map(|&value| value * (1.0 - value)).collect())
                    .collect();
                Some(hadamard(&back, &sigmoid_derivative))
            } else {
                None
            };

            //update old parameters wiht new optimized parameters
            //scaled by some learning rate learning_rate
            model[1].weight_matrix = matrix_subtraction(
                &model[1].weight_matrix,
                &scalar_multiplication(&learning_rate, &weight_gradient),
            );
            model[1].bias_matrix = matrix_subtraction(
                &model[1].bias_matrix,
                &scalar_multiplication(&learning_rate, &bias_gradient),
            );

            if let Some(d) = next_delta {
                delta = d;
            }
        }
    }

    pub fn shuffle_dataset(
        training_dataset: &Vec<Vec<f64>>,
        actual_value_dataset: &Vec<Vec<f64>>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        //either training_dataset[0].len() or actual_value_dataset.len() works
        let mut indices: Vec<usize> = (0..training_dataset.len()).collect();
        indices.shuffle(&mut rand::rng());

        let mut shuffled_training_dataset: Vec<Vec<f64>> =
            vec![vec![0.0; training_dataset[0].len()]; training_dataset.len()];
        let mut shuffled_actual_value_dataset: Vec<Vec<f64>> =
            vec![vec![0.0; actual_value_dataset[0].len()]; actual_value_dataset.len()];
        for i in 0..indices.len() {
            shuffled_training_dataset[i] = training_dataset[indices[i]].clone();
            shuffled_actual_value_dataset[i] = actual_value_dataset[indices[i]].clone();
        }

        (shuffled_training_dataset, shuffled_actual_value_dataset)
    }
}
