pub trait MachineLearning<T: std::fmt::Debug + Clone + std::fmt::Display>{
    fn init_model(model: T){}

    fn eval_model(model: T){}
}

pub mod neural_network {
    use crate::housing_price_dataset::generate_training_dataset;
    use crate::numpy::functions as function;
    use crate::numpy::matrix_arithmetic as array;
    use super::*;

    impl <T: std::fmt::Debug + Clone + std::fmt::Display> MachineLearning<T> for Vec<Layer> {
        fn init_model(model: T) {}

        fn eval_model(model: T) {}
    }

    #[derive(Debug)]
    pub struct Layer {
        pub neurons: usize,
        pub input_matrix: Vec<Vec<f64>>,
        pub weight_matrix: Vec<Vec<f64>>,
        pub bias_matrix: Vec<Vec<f64>>,
        pub weighted_sum_matrix: Vec<Vec<f64>>,
    }

    impl Layer {
        pub fn new(input_matrix: Vec<Vec<f64>>, neurons: usize) -> Self {
            let weight_matrix: Vec<Vec<f64>> =
                array::create_matrix(input_matrix[0].len(), neurons, true);
            let bias_matrix: Vec<Vec<f64>> = array::create_matrix(1, neurons, true);
            let weighted_sum_matrix =
                weighted_sum_matrix(&input_matrix, &weight_matrix, &bias_matrix);

            Layer {
                neurons,
                input_matrix,
                weight_matrix,
                bias_matrix,
                weighted_sum_matrix,
            }
        }

        pub fn forward(&mut self, input_matrix: &Vec<Vec<f64>>) {
            self.input_matrix = input_matrix.clone();
            self.weighted_sum_matrix =
                weighted_sum_matrix(&self.input_matrix, &self.weight_matrix, &self.bias_matrix);
        }
    }

    pub enum ActivationFunction {
        Sigmoid,
        Softmax,
    }
    pub struct ModelSettings {
        //number_of_neurons_hidden_layer.len() == number_of_hidden_layers
        pub number_of_neurons_hidden_layer: Vec<usize>,
        pub number_of_hidden_layers: usize,
        pub learning_rate: f64,
        //batch size must be less than or equal to number_of_datasets
        pub batch_size: usize,
        pub epochs: usize,
    }

    pub struct Dataset {
        pub number_of_datasets: usize,
        pub dataset: (Vec<Vec<f64>>, Vec<Vec<f64>>),
    }

    impl Dataset {
        pub fn new(number_of_datasets: usize) -> Self {
            Dataset {
                //field name matches argument name, so no need to explicitly specify the field name
                number_of_datasets,
                dataset: (Vec::new(), Vec::new()),
            }
        }

        pub fn generate_dataset(&mut self, dataset_type: AvailableDatasets) {
            match dataset_type {
                AvailableDatasets::Housing => {
                    self.dataset = generate_training_dataset(self.number_of_datasets);
                }
                _ => (),
            }
        }
    }

    pub enum AvailableDatasets {
        Housing,
    }

    pub fn init_input_layer(training_data: &Vec<Vec<f64>>) -> Layer {
        Layer::new(training_data.clone(), training_data[0].len())
    }

    pub fn eval_model_training(cost_history: &Vec<f64>) {
        dbg!(&cost_history);

        let cost_difference: f64 = cost_history[0] - cost_history[cost_history.len() - 1];

        println!("Cost Start: {}", cost_history[0]);
        println!("Cost End: {}", cost_history[cost_history.len() - 1]);
        println!("Cost Difference: {}", cost_difference);
    }

    pub fn train_model(
        data: &(Vec<Vec<f64>>, Vec<Vec<f64>>),
        model: &mut Vec<Layer>,
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
    ) -> Vec<f64> {
        assert!(batch_size > 0 && batch_size <= data.0.len());
        let mut cost_history: Vec<f64> = Vec::new();

        for epoch in 0..epochs {
            let (shuffled_training_dataset, shuffled_actual_value_dataset): (
                Vec<Vec<f64>>,
                Vec<Vec<f64>>,
            ) = function::shuffle_dataset(&data.0, &data.1);
            let (mut total, mut batches): (f64, usize) = (0.0, 0);

            for start in (0..shuffled_training_dataset.len()).step_by(batch_size) {
                let edge = (start + batch_size).min(shuffled_training_dataset.len());
                let training_batch = shuffled_training_dataset[start..edge].to_vec();
                let actual_batch = shuffled_actual_value_dataset[start..edge].to_vec();

                let prediction = forward_pass(model, &training_batch);

                total += function::mse(&prediction, &actual_batch);
                backward_pass(model, &prediction, &actual_batch, learning_rate);
                batches += 1;
            }

            let avg: f64 = total / batches as f64;
            cost_history.push(avg);

            if epoch % 100 == 0 {
                println!("Epoch: {} Cost: {}", epoch, avg);
            }
        }
        cost_history
    }

    pub fn init_model(
        number_of_neurons_hidden_layer: &Vec<usize>,
        input_layer: Layer,
        number_of_hidden_layers: &usize,
    ) -> Vec<Layer> {
        assert_eq!(
            number_of_neurons_hidden_layer.len(),
            number_of_hidden_layers.clone()
        );
        let mut model = init_hidden_layer(
            number_of_neurons_hidden_layer,
            input_layer,
            number_of_hidden_layers,
        );
        let output_layer: Layer = Layer::new(
            apply_activation_function(
                ActivationFunction::Sigmoid,
                model[model.len() - 1].weighted_sum_matrix.clone(),
            ),
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
                apply_activation_function(
                    ActivationFunction::Sigmoid,
                    layers[layers.len() - 1].weighted_sum_matrix.clone(),
                ),
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
        array::add_bias(
            &array::matrix_multiplication(input_matrix, weight_matrix),
            bias_matrix,
        )
    }

    pub fn apply_activation_function(
        activation_type: ActivationFunction,
        weighted_sum_matrix: Vec<Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        let mut res: Vec<Vec<f64>> = Vec::new();
        match activation_type {
            ActivationFunction::Sigmoid => {
                res = weighted_sum_matrix
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|value| function::sigmoid(*value))
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            }
            ActivationFunction::Softmax => {
                res = function::softmax(&weighted_sum_matrix);
            }
            _ => (),
        }
        res
    }

    pub fn forward_pass(model: &mut Vec<Layer>, batch: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut input = batch.clone();
        let n: usize = model.len() - 1;
        for (i, layer) in model.iter_mut().enumerate() {
            layer.forward(&input);
            //if we are at the last layer, the input matrix will be the weighted sum matrix
            input = if i == n {
                layer.weighted_sum_matrix.clone()
            } else {
                //new input layer is just the weighted sum matrix after a activation fucntion of the previous layer
                apply_activation_function(
                    ActivationFunction::Sigmoid,
                    layer.weighted_sum_matrix.clone(),
                )
            };
        }
        let prediction = input.clone();
        prediction
    }

    fn backward_pass(
        model: &mut Vec<Layer>,
        prediction: &Vec<Vec<f64>>,
        actual: &Vec<Vec<f64>>,
        learning_rate: f64,
    ) {
        function::stochastic_gradient_descent(model, prediction, actual, learning_rate);
    }

    pub fn train_test_split(
        training_dataset: &Vec<Vec<f64>>,
        actual_value_dataset: &Vec<Vec<f64>>,
        training_split: f64,
    ) -> (
        //x: training dataset, y: actual value dataset
        //(training split x, training split y)
        //(testing split x, testing split y)
        (Vec<Vec<f64>>, Vec<Vec<f64>>),
        (Vec<Vec<f64>>, Vec<Vec<f64>>),
    ) {
        let (shuffled_training_dataset, shuffled_actual_value_dataset): (
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
        ) = function::shuffle_dataset(training_dataset, actual_value_dataset);
        //rounds the float down (truncation)
        let split = (shuffled_training_dataset.len() as f64 * training_split) as usize;
        (
            //[..split] is exclusive of split
            (
                shuffled_training_dataset[..split].to_vec(),
                shuffled_actual_value_dataset[..split].to_vec(),
            ),
            //[split..] is inclusive of split
            (
                shuffled_training_dataset[split..].to_vec(),
                shuffled_actual_value_dataset[split..].to_vec(),
            ),
        )
    }

    pub fn test_model(
        //foward pass requires model to be a mutable reference
        model: &mut Vec<Layer>,
        //first element in the tuple is the testing training set and the second is the testing actual value set
        testing_split_data: &(Vec<Vec<f64>>, Vec<Vec<f64>>),
        min_price: f64,
        max_price: f64,
        margin_of_error: f64,
    ) {
        let prediction = forward_pass(model, &testing_split_data.0);

        //let scaled_mse = function::mse(&prediction, &testing_split_data.1);

        let (mut squared_sum, mut abs_sum, mut correct_predictions): (f64, f64, f64) =
            (0.0, 0.0, 0.0);
        //prediction.iter(): Vec<Vec<f64>>
        //therefore prediction: Vec<f64> -> same case for actual_value and &testing_split_data.1
        for (prediction, actual_value) in prediction.iter().zip(&testing_split_data.1) {
            let prediction_price =
                crate::housing_price_dataset::unscale_price(&prediction[0], &min_price, &max_price);
            let actual_value_price = crate::housing_price_dataset::unscale_price(
                &actual_value[0],
                &min_price,
                &max_price,
            );
            let error = prediction_price - actual_value_price;
            squared_sum += error * error;
            abs_sum += error.abs();

            if error.abs() <= margin_of_error {
                correct_predictions += 1.0;
            }
        }
        //need it to be a f64 so no type issues when calculating metrics
        let number_of_predictions: f64 = prediction.len() as f64;
        //Root Mean Square Error
        let rmse: f64 = (squared_sum / number_of_predictions).sqrt();
        //Mean Absolute Error
        let mae: f64 = abs_sum / number_of_predictions;
        let accuracy: f64 = (correct_predictions / number_of_predictions) * 100.0;
        println!("Test samples: {}", prediction.len());
        println!("RMSE: {}", rmse);
        println!("MAE: {}", mae);
        println!("Margin of Error: ${}", margin_of_error);
        println!("Accuracy: {}%", accuracy);
    }
}

pub mod convolutional_neural_network {
    use crate::housing_price_dataset::generate_training_dataset;
    use crate::numpy::functions as function;
    use crate::numpy::matrix_arithmetic as array;
    use std::f64::{MAX, NEG_INFINITY};

    #[derive(Debug)]
    struct ConvolutionalLayer {
        //tensor inputs
        input: Vec<Vec<Vec<f64>>>,
        //multiple kernals per conv layer, will only be using square kernals
        kernals: Vec<Vec<Vec<f64>>>,
        stride: usize,
        bias_matrix: Vec<f64>,
        activation_function: ActivationFunctionCnn,
    }

    #[derive(Debug)]
    struct PoolingLayer {
        //will be using only squares as well
        pooling_window: usize,
        stride: usize,
        pooling_type: PoolingType,
    }

    #[derive(Debug)]
    struct ConvolutionalBlock {
        conv_layer: ConvolutionalLayer,
        pooling_layer: PoolingLayer,
    }

    //hey future me, if you ever want to make this code more presentable, remove the amount of cloning with references
    #[derive(Debug, Clone)]
    enum PoolingType {
        MaxPooling,
        AveragePooling,
    }

    #[derive(Debug, Clone)]
    enum ActivationFunctionCnn {
        ReLu,
        //wont impl the rest for now
        LeakyReLu,
        Elu,
        Tanh,
    }

    //intialize the layers, backpropgation will use the vec storing Convblocks with &mut
    impl ConvolutionalBlock {
        fn new(
            input: &Vec<Vec<Vec<f64>>>,
            number_of_kernals: usize,
            kernal_size: &usize,
            kernal_stride: &usize,
            activation_function: ActivationFunctionCnn,
            pooling_stride: usize,
            pooling_window: usize,
            pooling_type: PoolingType,
        ) -> Self {
            let mut kernals: Vec<Vec<Vec<f64>>> = Vec::new();
            for i in 0..number_of_kernals {
                //remember: fn push(&mut self)
                kernals.push(array::create_matrix(*kernal_size, *kernal_size, true));
            }

            let bias_matrix: Vec<f64> = array::cnn_bias_matrix(kernals.len());

            ConvolutionalBlock {
                conv_layer: ConvolutionalLayer {
                    //if i want to access the input after training for whatever reason
                    input: input.clone(),
                    kernals,
                    stride: *kernal_stride,
                    bias_matrix,
                    activation_function,
                },
                pooling_layer: PoolingLayer {
                    pooling_window,
                    stride: pooling_stride,
                    pooling_type,
                },
            }
        }

        fn foward(
            &mut self,
            prev_input: &Vec<Vec<Vec<f64>>>,
            activation_function: ActivationFunctionCnn,
        ) {
            match activation_function {
                ActivationFunctionCnn::ReLu => {
                    self.conv_layer.input = apply_activation_function_cnn(
                        activation_function,
                        &add_bias(
                            &create_feature_map(
                                &prev_input,
                                &self.conv_layer.kernals,
                                &self.conv_layer.stride,
                            ),
                            &self.conv_layer.bias_matrix,
                        ),
                    )
                }
                //the rest of the activation functions
                _ => (),
            };
        }
    }

    fn add_bias(feature_map: &Vec<Vec<Vec<f64>>>, bias: &Vec<f64>) -> Vec<Vec<Vec<f64>>> {
        assert!(feature_map.len() == bias.len());
        let mut res: Vec<Vec<Vec<f64>>> = Vec::new();

        res = (0..bias.len())
            .map(|index| {
                feature_map[index]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&pixel| pixel + bias[index])
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>();

        res
    }

    fn apply_activation_function_cnn(
        activation_type: ActivationFunctionCnn,
        input: &Vec<Vec<Vec<f64>>>,
    ) -> Vec<Vec<Vec<f64>>> {
        let mut res: Vec<Vec<Vec<f64>>> = Vec::new();
        match activation_type {
            ActivationFunctionCnn::ReLu => {
                res = input
                    .iter()
                    .map(|channel| {
                        channel
                            .iter()
                            .map(|row| {
                                row.iter()
                                    .map(|&pixel| function::relu(pixel))
                                    .collect::<Vec<f64>>()
                            })
                            .collect::<Vec<Vec<f64>>>()
                    })
                    .collect::<Vec<Vec<Vec<f64>>>>()
            }
            //the rest not implemented yet
            _ => (),
        };
        res
    }

    fn pool_feature_map(
        pooling_type: PoolingType,
        input: &Vec<Vec<Vec<f64>>>,
        pooling_stride: &usize,
        pooling_window: &usize,
    ) -> Vec<Vec<Vec<f64>>> {
        //we provide the guarentee that the pooling window is a square
        assert!(*pooling_stride > 0 && *pooling_window > 1);
        let mut new_feature_maps: Vec<Vec<Vec<f64>>> =
            vec![
                vec![
                    vec![0.0; (input[0][0].len() - pooling_window) / pooling_stride + 1];
                    (input[0].len() - pooling_window) / pooling_stride + 1
                ];
                input.len()
            ];

        match pooling_type {
            PoolingType::MaxPooling => {
                new_feature_maps = input
                    .iter()
                    .map(|channel| {
                        (0..new_feature_maps.len())
                            .map(|i| {
                                (0..new_feature_maps[0].len())
                                    .map(|j| {
                                        channel[i * pooling_stride
                                            ..i * pooling_stride + pooling_window]
                                            //flat_map -> flatten + map
                                            .iter()
                                            .flat_map(|row| {
                                                &row[j * pooling_stride
                                                    ..j * pooling_stride + pooling_window]
                                            })
                                            //iter() has ref entries, .copied() has owned entries
                                            .copied()
                                            .fold(f64::NEG_INFINITY, f64::max)
                                    })
                                    .collect::<Vec<f64>>()
                            })
                            .collect::<Vec<Vec<f64>>>()
                    })
                    .collect::<Vec<Vec<Vec<f64>>>>()
            }
            PoolingType::AveragePooling => {
                let n: f64 = pooling_window.pow(2) as f64;
            }
            _ => {}
        }
        //used as the input for the next conv block
        new_feature_maps
    }

    pub fn init_cnn(
        images: &Vec<Vec<Vec<f64>>>,
        number_of_conv_blocks: usize,
        number_of_kernals: usize,
        kernal_size: Vec<usize>,
        kernal_stride: Vec<usize>,
        activation_type: ActivationFunctionCnn,
        pooling_window: usize,
        pooling_stride: usize,
        pooling_type: PoolingType,
    ) -> Vec<ConvolutionalBlock> {
        //the indices of kernal_size and kernal_stride represent their respective kernal sizes and strides at that convolutional block
        let mut convolutional_base: Vec<ConvolutionalBlock> = Vec::new();
        //base, the next convolutional blocks will have different inputs
        convolutional_base.push(ConvolutionalBlock::new(
            images,
            number_of_kernals,
            &kernal_size[0],
            &kernal_stride[0],
            activation_type.clone(),
            pooling_stride,
            pooling_window,
            pooling_type.clone(),
        ));

        for i in 1..number_of_conv_blocks {
            let dummy_input: Vec<Vec<Vec<f64>>> = Vec::new();
            convolutional_base.push(ConvolutionalBlock::new(
                &dummy_input,
                number_of_kernals,
                &kernal_size[i],
                &kernal_stride[i],
                activation_type.clone(),
                pooling_stride,
                pooling_window,
                pooling_type.clone(),
            ));
        }

        convolutional_base
    }

    //forward pass will iterate over model_cnn and call forward at each iteration
    //forward_pass will return the last output of which we will later flatten and feed into the neural net
    //currently, the conv part is fully made, next step is training
    fn cnn_forward_pass(
        model_cnn: &mut Vec<ConvolutionalBlock>,
        batch: &Vec<Vec<Vec<f64>>>,
    ) -> Vec<Vec<Vec<f64>>> {
    }

    fn cnn_backward_pass() {}

    fn forward_pass() {}

    fn backward_pass() {}

    fn create_feature_map(
        input: &Vec<Vec<Vec<f64>>>,
        kernals: &Vec<Vec<Vec<f64>>>,
        stride: &usize,
    ) -> Vec<Vec<Vec<f64>>> {
        assert!(stride.clone() > 0 as usize);

        let mut feature_maps: Vec<Vec<Vec<f64>>> =
            vec![
                vec![
                    vec![0.0; (input[0][0].len() - kernals[0][0].len()) / stride + 1];
                    (input[0].len() - kernals[0].len()) / stride + 1
                ];
                input.len()
            ];

        //input.len() == kernals.len()
        input
            .iter()
            .enumerate()
            .map(|(i, channel)| {
                (0..feature_maps[0].len())
                    .map(|j| {
                        (0..feature_maps[0][0].len())
                            .map(|k| {
                                element_wise_dot_product(
                                    channel,
                                    &kernals[i],
                                    j * stride,
                                    k * stride,
                                )
                            })
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>()
    }

    fn element_wise_dot_product(
        input: &Vec<Vec<f64>>,
        kernal: &Vec<Vec<f64>>,
        row_index_after_stride: usize,
        col_index_after_stride: usize,
    ) -> f64 {
        let mut dot_product: f64 = 0.0;

        for i in 0..kernal.len() {
            for j in 0..kernal[0].len() {
                dot_product +=
                    kernal[i][j] * input[i + row_index_after_stride][j + col_index_after_stride];
            }
        }
        dot_product
    }
}
