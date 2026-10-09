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
            .map(|row| row.iter().zip(&bias_matrix[0]).map(|(a, b)| a + b).collect())
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
        let mut transposed_matrix: Vec<Vec<f64>> = vec![vec![0.0; matrix[0].len()]; matrix.len()];

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
}

pub mod functions {
    //for scope reasons
    use super::matrix_arithmetic::*;
    use crate::housing_price_dataset::scale_price;
    use rand::seq::SliceRandom;
    use rand::*;
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

    pub fn stochastic_gradient_descent(
        mut bias_matrix: Vec<Vec<f64>>,
        mut weight_matrix: Vec<Vec<f64>>,
        data: &(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
    ) -> ((Vec<Vec<f64>>, Vec<Vec<f64>>), Vec<f64>) {
        let mut cost_history: Vec<f64> = Vec::new();

        for epoch in 0..epochs {
            let (shuffled_training_dataset, shuffled_actual_value_dataset) =
                shuffle_dataset(&data.0, &data.1);

            for i in (0..data.0[0].len()).step_by(batch_size) {
                dbg!(data.0.len());
                //batch_size + i properly increments the slice
                //fix: slice if the mod isnt 0, **get the remainder**
                let remainder = (i + batch_size).min(shuffled_training_dataset.len());
                let training_batch: Vec<Vec<f64>> =
                    shuffled_training_dataset[0][i..remainder].to_vec();
                let actual_value_batch: Vec<Vec<f64>> =
                    shuffled_actual_value_dataset[i..remainder].to_vec();

                //take the partial of w_i for hat(y_i)
                //take the partial of b_i for hat(y_i)
                dbg!(&weight_matrix);
                dbg!(&bias_matrix);
                let (weight_gradient, bias_gradient): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (
                    scalar_multiplication(
                        &(2.0 / batch_size as f64),
                        &matrix_multiplication(
                            &transpose(&training_batch),
                            &matrix_subtraction(
                                &add_bias(
                                    &matrix_multiplication(&training_batch, &weight_matrix),
                                    &bias_matrix,
                                ),
                                &actual_value_batch,
                            ),
                        ),
                    ),
                    scalar_multiplication(
                        &(2.0 / batch_size as f64),
                        &sum_columns(&matrix_subtraction(
                            &add_bias(
                                &matrix_multiplication(&training_batch, &weight_matrix),
                                &bias_matrix,
                            ),
                            &actual_value_batch,
                        )),
                    ),
                );

                let updated_weight_matrix = matrix_subtraction(
                    &weight_matrix,
                    &scalar_multiplication(&learning_rate, &weight_gradient),
                );
                let updated_bias_matrix = matrix_subtraction(
                    &bias_matrix,
                    &scalar_multiplication(&learning_rate, &bias_gradient),
                );

                weight_matrix = updated_weight_matrix;
                bias_matrix = updated_bias_matrix;

                let prediction: Vec<Vec<f64>> = add_bias(
                    &matrix_multiplication(&training_batch, &weight_matrix),
                    &bias_matrix,
                );

                let cost: f64 = MSE(&prediction, &actual_value_batch);

                cost_history.push(cost);

                if epoch % 100 == 0 {
                    println!("Epoch: {} Cost: {}", epoch, cost)
                }
            }
        }

        ((weight_matrix, bias_matrix), cost_history)
    }

    pub fn shuffle_dataset(
        training_dataset: &Vec<Vec<Vec<f64>>>,
        actual_value_dataset: &Vec<Vec<f64>>,
    ) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        //either training_dataset[0].len() or actual_value_dataset.len() works
        let mut indices: Vec<usize> = (0..training_dataset[0].len()).collect();
        indices.shuffle(&mut rand::rng());

        let mut shuffled_training_dataset: Vec<Vec<Vec<f64>>> =
            vec![vec![
                vec![0.0; training_dataset[0][0].len()];
                training_dataset[0].len()
            ]];
        let mut shuffled_actual_value_dataset: Vec<Vec<f64>> =
            vec![vec![0.0; actual_value_dataset[0].len()]; actual_value_dataset.len()];
        for i in 0..indices.len() {
            shuffled_training_dataset[0][i] = training_dataset[0][indices[i]].clone();
            shuffled_actual_value_dataset[i] = actual_value_dataset[indices[i]].clone();
        }

        (shuffled_training_dataset, shuffled_actual_value_dataset)
    }
}
