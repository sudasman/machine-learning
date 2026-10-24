pub trait MachineLearning<T: std::fmt::Debug + Clone + std::fmt::Display> {
    fn init_model(model: T) {}

    fn train_model(model: T) {}

    fn test_model(model: T) {}

    fn eval_model(model: T) {}
}

#[derive(Debug, Clone)]
pub enum Models {
    NeuralNetwork,
    ConvolutionalNeuralNetwork,
}

pub enum AvailableDatasets {
    Housing,
    Numbers,
}

pub mod neural_network {
    use super::*;
    use crate::housing_price_dataset::generate_training_dataset;
    use crate::numpy::functions as function;
    use crate::numpy::matrix_arithmetic as array;

    impl<T: std::fmt::Debug + Clone + std::fmt::Display> MachineLearning<T> for Vec<Layer> {
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

    #[derive(Debug, Clone)]
    pub enum ActivationFunction {
        Sigmoid,
        Softmax,
    }

    #[derive(Debug, Clone)]
    pub enum LossFunction {
        Mse,
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
        loss_function: Option<LossFunction>,
        loss_function_cnn: Option<convolutional_neural_network::LossFunctionCnn>,
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

                let prediction = forward_pass(model, &training_batch, &ActivationFunction::Sigmoid);

                total += function::use_loss_function(
                    (&prediction, &actual_batch),
                    &loss_function,
                    &loss_function_cnn,
                );
                let delta = array::scalar_multiplication(
                    &(2.0 / prediction.len() as f64),
                    &array::matrix_subtraction(&prediction, &actual_batch),
                );
                let _ = backward_pass(model, delta, learning_rate);
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
        dataset_type: &AvailableDatasets,
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

        match dataset_type {
            AvailableDatasets::Housing => {
                let output_layer: Layer = Layer::new(
                    apply_activation_function(
                        ActivationFunction::Sigmoid,
                        model[model.len() - 1].weighted_sum_matrix.clone(),
                    ),
                    1,
                );
                model.push(output_layer);
            }
            AvailableDatasets::Numbers => {
                let output_layer: Layer = Layer::new(
                    apply_activation_function(
                        ActivationFunction::Sigmoid,
                        model[model.len() - 1].weighted_sum_matrix.clone(),
                    ),
                    10,
                );
                model.push(output_layer);
            }
        }

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

    pub fn forward_pass(
        model: &mut Vec<Layer>,
        batch: &Vec<Vec<f64>>,
        activation_function: &ActivationFunction,
    ) -> Vec<Vec<f64>> {
        let mut input = batch.clone();
        let n: usize = model.len() - 1;
        for (i, layer) in model.iter_mut().enumerate() {
            layer.forward(&input);
            //if we are at the last layer, the input matrix will be the weighted sum matrix
            input = if i == n {
                match activation_function {
                    ActivationFunction::Softmax => apply_activation_function(
                        ActivationFunction::Softmax,
                        layer.weighted_sum_matrix.clone(),
                    ),
                    _ => layer.weighted_sum_matrix.clone(),
                }
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

    pub fn backward_pass(
        model: &mut Vec<Layer>,
        delta: Vec<Vec<f64>>,
        learning_rate: f64,
    ) -> Vec<Vec<f64>> {
        function::gradient_descent(model, delta, learning_rate)
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
        let prediction = forward_pass(model, &testing_split_data.0, &ActivationFunction::Sigmoid);

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

//im already too deep, the kernel misspelling will be fixed at the end
pub mod convolutional_neural_network {
    use super::*;
    use crate::housing_price_dataset::generate_training_dataset;
    use crate::numpy::functions as function;
    use crate::numpy::matrix_arithmetic as array;
    use std::f64::{MAX, NEG_INFINITY};

    impl<T: std::fmt::Debug + Clone + std::fmt::Display> MachineLearning<T>
        for Vec<ConvolutionalBlock>
    {
        fn init_model(model: T) {}

        fn eval_model(model: T) {}
    }

    #[derive(Debug)]
    pub struct ConvolutionalLayer {
        //tensor inputs
        input: Vec<Vec<Vec<f64>>>,
        //multiple kernels per conv layer, will only be using square kernels
        //[number of kernels][channel][row][col]
        kernels: Vec<Vec<Vec<Vec<f64>>>>,
        stride: usize,
        bias_matrix: Vec<f64>,
        activation_function: ActivationFunctionCnn,
    }

    #[derive(Debug)]
    pub struct PoolingLayer {
        //will be using only squares as well
        pooling_window: usize,
        pooling_stride: usize,
        pooling_type: PoolingType,
    }

    #[derive(Debug)]
    pub struct ConvolutionalBlock {
        conv_layer: ConvolutionalLayer,
        pooling_layer: PoolingLayer,
        kernel_gradient: Vec<Vec<Vec<Vec<f64>>>>,
        bias_gradient: Vec<f64>,
    }

    //used for optimization
    struct BlockCache {
        prev_input: Vec<Vec<Vec<f64>>>,
        //same as pre-pool
        post_activation: Vec<Vec<Vec<f64>>>,
    }

    //hey future me, if you ever want to make this code more presentable, remove the amount of cloning with references
    #[derive(Debug, Clone)]
    pub enum PoolingType {
        MaxPooling,
        AveragePooling,
    }

    #[derive(Debug, Clone)]
    pub enum ActivationFunctionCnn {
        ReLu,
        //wont impl the rest for now
        LeakyReLu,
        Elu,
        Tanh,
    }

    #[derive(Debug, Clone)]
    pub enum LossFunctionCnn {
        CrossEntropyLoss,
    }

    //intialize the layers, backpropgation will use the vec storing Convblocks with &mut
    //note: padding not implemented
    //neither is a fully rgb input
    //idea for rgb -> use tuple to represent rgb where every pixel is a 3 tuple
    impl ConvolutionalBlock {
        fn new(
            input: &Vec<Vec<Vec<f64>>>,
            number_of_kernels: &usize,
            kernel_channels: &usize,
            kernel_size: &usize,
            kernel_stride: &usize,
            activation_function: &ActivationFunctionCnn,
            pooling_stride: &usize,
            pooling_window: &usize,
            pooling_type: &PoolingType,
        ) -> Self {
            let kernels = (0..*number_of_kernels)
                .map(|_| {
                    (0..*kernel_channels)
                        .map(|_| {
                            //remember: fn push(&mut self)
                            array::create_matrix(kernel_size.clone(), kernel_size.clone(), true)
                        })
                        .collect::<Vec<Vec<Vec<f64>>>>()
                })
                .collect::<Vec<Vec<Vec<Vec<f64>>>>>();

            let bias_matrix: Vec<f64> = array::cnn_bias_matrix(kernels.len());
            let n = bias_matrix.len();

            ConvolutionalBlock {
                conv_layer: ConvolutionalLayer {
                    //if i want to access the input after training for whatever reason
                    input: input.clone(),
                    kernels: (*kernels).to_vec(),
                    stride: *kernel_stride,
                    bias_matrix: (*bias_matrix).to_vec(),
                    activation_function: activation_function.clone(),
                },
                pooling_layer: PoolingLayer {
                    pooling_window: *pooling_window,
                    pooling_stride: *pooling_stride,
                    pooling_type: pooling_type.clone(),
                },
                kernel_gradient: vec![
                    vec![
                        vec![vec![0.0; *kernel_size]; *kernel_size];
                        *kernel_channels
                    ];
                    *number_of_kernels
                ],
                bias_gradient: vec![0.0; n],
            }
        }

        fn forward(
            &mut self,
            prev_input: &Vec<Vec<Vec<f64>>>,
            activation_function: &ActivationFunctionCnn,
        ) -> (Vec<Vec<Vec<f64>>>, BlockCache) {
            let mut activation_function_applied: Vec<Vec<Vec<f64>>> = Vec::new();
            let mut pooled: Vec<Vec<Vec<f64>>> = Vec::new();
            match activation_function {
                ActivationFunctionCnn::ReLu => {
                    activation_function_applied = apply_activation_function_cnn(
                        activation_function,
                        &add_bias(
                            &create_feature_map(
                                prev_input,
                                &self.conv_layer.kernels,
                                &self.conv_layer.stride,
                            ),
                            &self.conv_layer.bias_matrix,
                        ),
                    );
                    pooled = pool_feature_map(
                        &self.pooling_layer.pooling_type,
                        &activation_function_applied,
                        &self.pooling_layer.pooling_stride,
                        &self.pooling_layer.pooling_window,
                    );
                    self.conv_layer.input = pooled.clone();
                }
                //the rest of the activation functions
                _ => (),
            };

            (
                self.conv_layer.input.clone(),
                BlockCache {
                    prev_input: prev_input.clone(),
                    post_activation: activation_function_applied,
                },
            )
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
        activation_type: &ActivationFunctionCnn,
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

    //used as the input for the next conv block
    fn pool_feature_map(
        pooling_type: &PoolingType,
        input: &Vec<Vec<Vec<f64>>>,
        pooling_stride: &usize,
        pooling_window: &usize,
    ) -> Vec<Vec<Vec<f64>>> {
        //we provide the guarentee that the pooling window is a square
        assert!(*pooling_stride > 0 && *pooling_window > 1);

        match pooling_type {
            PoolingType::MaxPooling => {
                input
                    .iter()
                    .map(|channel| {
                        (0..(input[0].len() - pooling_window) / pooling_stride + 1)
                            .map(|i| {
                                (0..(input[0][0].len() - pooling_window) / pooling_stride + 1)
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
                unimplemented!();
            }
        }
    }

    pub fn init_cnn(
        images: &Vec<Vec<Vec<f64>>>,
        number_of_conv_blocks: &usize,
        number_of_kernels: &Vec<usize>,
        kernel_size: &Vec<usize>,
        kernel_stride: &Vec<usize>,
        activation_type: &ActivationFunctionCnn,
        pooling_window: &usize,
        pooling_stride: &usize,
        pooling_type: &PoolingType,
    ) -> Vec<ConvolutionalBlock> {
        //the indices of kernel_size and kernel_stride represent their respective kernel sizes and strides at that convolutional block
        let mut convolutional_base: Vec<ConvolutionalBlock> = Vec::new();
        let mut input_channels = images.len();
        //base, the next convolutional blocks will have different inputs
        convolutional_base.push(ConvolutionalBlock::new(
            &images,
            &number_of_kernels[0],
            &input_channels,
            &kernel_size[0],
            &kernel_stride[0],
            &activation_type,
            &pooling_stride,
            &pooling_window,
            &pooling_type,
        ));
        input_channels = number_of_kernels[0];
//m
        for i in 1..*number_of_conv_blocks {
            let dummy_input: Vec<Vec<Vec<f64>>> = Vec::new();
            convolutional_base.push(ConvolutionalBlock::new(
                &dummy_input,
                &number_of_kernels[i],
                &input_channels,
                &kernel_size[i],
                &kernel_stride[i],
                &activation_type,
                &pooling_stride,
                &pooling_window,
                &pooling_type,
            ));
            input_channels = number_of_kernels[i];
        }

        convolutional_base
    }

    fn cnn_forward_pass(
        convolutional_base: &mut Vec<ConvolutionalBlock>,
        train_input: &Vec<Vec<Vec<f64>>>,
        activation_function: &ActivationFunctionCnn,
    ) -> (Vec<Vec<Vec<f64>>>, Vec<BlockCache>) {
        let mut input = train_input.clone();
        let mut temp: Vec<BlockCache> = Vec::new();

        for conv_block in convolutional_base.iter_mut() {
            let (output, cache) = conv_block.forward(&input, activation_function);
            temp.push(cache);
            input = output;
        }

        let cnn_output = input.clone();
        (cnn_output, temp)
    }

    fn flatten(output: &Vec<Vec<Vec<f64>>>) -> Vec<f64> {
        output
            .iter()
            .flatten()
            .flatten()
            .cloned()
            .collect::<Vec<f64>>()
    }

    //pass model.1 as Vec::new() -> initlize in this fn
    fn train_model_cnn(
        images: &Vec<Vec<Vec<f64>>>,
        labels: &Vec<Vec<f64>>,
        model: &mut (Vec<ConvolutionalBlock>, Vec<neural_network::Layer>),
        model_settings: neural_network::ModelSettings,
        loss_function: Option<LossFunctionCnn>,
    ) {
        assert!(model_settings.batch_size > 0 && model_settings.batch_size <= images.len());
        let mut cost_history: Vec<f64> = Vec::new();
        let epochs: usize = model_settings.epochs.clone();
        //[image][conv_block][cache]
        for epoch in 0..epochs {
            let mut images_cache: Vec<Vec<BlockCache>> = Vec::new();
            let (shuffled_training_dataset, shuffled_labels_dataset): (
                Vec<Vec<Vec<f64>>>,
                Vec<Vec<f64>>,
            ) = function::shuffle_dataset_tensor(images, labels);
            let (mut total, mut batches): (f64, usize) = (0.0, 0);

            for start in
                (0..shuffled_training_dataset.len()).step_by(model_settings.batch_size.clone())
            {
                let mut flattened_images: Vec<Vec<f64>> = Vec::new();
                let edge = (start + model_settings.batch_size.clone())
                    .min(shuffled_training_dataset.len());

                let training_batch = shuffled_training_dataset[start..edge].to_vec();
                let label_batch = shuffled_labels_dataset[start..edge].to_vec();

                let mut output_dim: Vec<(usize, usize, usize)> = Vec::new();
                for image in &training_batch {
                    //passed in as [channel][row][col], where channel == 1
                    let (cnn_output, cache): (Vec<Vec<Vec<f64>>>, Vec<BlockCache>) =
                        cnn_forward_pass(
                            &mut model.0,
                            &vec![image.clone()],
                            &ActivationFunctionCnn::ReLu,
                        );
                    images_cache.push(cache);
                    output_dim.push((
                        cnn_output.len(),
                        cnn_output[0].len(),
                        cnn_output[0][0].len(),
                    ));
                    //flatten -> Vec<f64>
                    flattened_images.push(flatten(&cnn_output));
                }

                let prediction: Vec<Vec<f64>> = neural_network::forward_pass(
                    &mut model.1,
                    &flattened_images,
                    &neural_network::ActivationFunction::Softmax,
                );

                total +=
                    function::use_loss_function((&prediction, &label_batch), &None, &loss_function);

                let delta = softmax_cross_entropy_grad(&prediction, &label_batch);

                let extract_delta: Vec<Vec<f64>> = neural_network::backward_pass(
                    &mut model.1,
                    delta,
                    model_settings.learning_rate.clone(),
                );

                // let grad: Vec<Vec<Vec<Vec<f64>>>> = unflatten(&extract_delta);

                // cnn_backward_pass(&mut model.0, &extract_delta, &model_settings.learning_rate);
            }
        }
    }

    pub fn gradient_descent_cnn(
        model_cnn: &mut Vec<ConvolutionalBlock>,
        //[number_of_delta][channel][row][col]
        mut delta: &Vec<Vec<Vec<Vec<f64>>>>,
        learning_rate: &f64,
    ) {
        //conv block stores the current feature map and the previous feature map before pooling,activation function, and bias
        //convolutional_base[i].conv_layer.input is also thought of as convolutional_base[i + 1].prev_input
    }

    fn max_pool_backpass(
        pre_pool: &Vec<Vec<Vec<f64>>>,
        delta: &Vec<Vec<Vec<f64>>>,
        pooling_window: &usize,
        pooling_stride: &usize,
    ) -> Vec<Vec<Vec<f64>>> {
        let mut grad_pre_pool =
            vec![vec![vec![0.0; pre_pool[0][0].len()]; pre_pool[0].len()]; pre_pool.len()];

        for channel in 0..grad_pre_pool.len() {
            for i in 0..delta[channel].len() {
                for j in 0..delta[channel][i].len() {
                    let (mut max, mut pos): (f64, (usize, usize)) = (f64::NEG_INFINITY, (0, 0));
                    for u in 0..pooling_window.clone() {
                        for v in 0..*pooling_window {
                            let (row, col) = (i * pooling_stride + u, j * pooling_stride + v);
                            max = if max < pre_pool[channel][row][col] {
                                pos = (row, col);
                                pre_pool[channel][row][col]
                            } else {
                                max
                            }
                        }
                    }
                    grad_pre_pool[channel][pos.0][pos.1] += delta[channel][i][j];
                }
            }
        }
        grad_pre_pool
    }

    fn conv_backward() {}

    fn cnn_backward_pass() {}

    //zero the grad during training
    fn cnn_zero_grad() {}

    fn cnn_update_parameters() {}

    fn grad_check() {}

    pub fn softmax_cross_entropy_grad(
        predictions: &Vec<Vec<f64>>,
        labels: &Vec<Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        let n: f64 = predictions.len() as f64;

        predictions
            .iter()
            .zip(labels)
            .map(|(row, label)| {
                row.iter()
                    .enumerate()
                    //labels.[0].len() == 1
                    //labels stores 1d entries of the actual number
                    //i represents the index of the prediction in predictions -> predictions.len() == 10
                    //if i (the prediction) == label[0] (the actual value)
                    .map(|(i, &prediction)| {
                        (prediction - if i == label[0] as usize { 1.0 } else { 0.0 }) / n
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
    }

    fn unflatten(
        input: &Vec<Vec<f64>>,
        (channels, rows, cols): (usize, usize, usize),
    ) -> Vec<Vec<Vec<Vec<f64>>>> {
        input
            .iter()
            .map(|image| unflatten_single(&image, (channels, rows, cols)))
            .collect::<Vec<Vec<Vec<Vec<f64>>>>>()
    }

    //note to self: <T: ?Sized> (input: T)
    fn unflatten_single(
        input: &[f64],
        (channels, rows, cols): (usize, usize, usize),
    ) -> Vec<Vec<Vec<f64>>> {
        assert_eq!(input.len(), channels * rows * cols);
        (0..channels)
            .map(|c| {
                (0..rows)
                    .map(|r| input[(c * rows + r) * cols..(c * rows + r + 1) * cols].to_vec())
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>()
    }

    //leetcode flashbacks
    //there may or may not be a error here
    fn create_feature_map(
        //[channel][row][col]
        input: &Vec<Vec<Vec<f64>>>,
        //[number_of_kernels][channel][row][col]
        kernels: &Vec<Vec<Vec<Vec<f64>>>>,
        stride: &usize,
    ) -> Vec<Vec<Vec<f64>>> {
        assert!(stride.clone() > 0 as usize && kernels.len() > 0);
        assert!(input.len() == kernels[0].len());

        kernels
            .iter()
            .map(|kernel| {
                (0..(input[0].len() - kernels[0][0].len()) / stride + 1)
                    .map(|i| {
                        (0..(input[0][0].len() - kernels[0][0][0].len()) / stride + 1)
                            .map(|j| {
                                input
                                    .iter()
                                    .zip(kernel)
                                    .map(|(channel, kernel)| {
                                        element_wise_dot_product(
                                            channel,
                                            kernel,
                                            i * stride,
                                            j * stride,
                                        )
                                    })
                                    .sum::<f64>()
                            })
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>()
    }

    fn element_wise_dot_product(
        input: &Vec<Vec<f64>>,
        kernel: &Vec<Vec<f64>>,
        row_index_after_stride: usize,
        col_index_after_stride: usize,
    ) -> f64 {
        let mut dot_product: f64 = 0.0;

        for i in 0..kernel.len() {
            for j in 0..kernel[0].len() {
                dot_product +=
                    kernel[i][j] * input[i + row_index_after_stride][j + col_index_after_stride];
            }
        }
        dot_product
    }
}
