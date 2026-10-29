use crate::ml::{self, convolutional_neural_network};
use crate::number_dataset;

pub fn new() -> (
    (
        Vec<convolutional_neural_network::ConvolutionalBlock>,
        Vec<ml::neural_network::Layer>,
    ),
    f64,
) {
    //training_set = (training_images, training_labels)
    //testing_set = (testing_images, testing_labels)
    let (mut training_set, mut testing_set): (
        (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
        (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
    ) = number_dataset::load_dataset();

    // dbg!(&training_set.1);
    // number_dataset::show_sample_aimage(&training_set.0, &training_set.1, 0, Some(7));

    let model_settings: ml::neural_network::ModelSettings = ml::neural_network::ModelSettings {
        number_of_neurons_hidden_layer: vec![32, 16],
        number_of_hidden_layers: 2,
        learning_rate: 0.1,
        batch_size: 10,
        epochs: 20,
    };

    let cnn_model_settings: convolutional_neural_network::cnn_model_settings =
        convolutional_neural_network::cnn_model_settings {
            number_of_conv_blocks: 1,
            //index corresponds to each block
            number_of_kernels: vec![8],
            kernel_size: vec![3],
            kernel_stride: vec![3],
            activation_type: convolutional_neural_network::ActivationFunctionCnn::ReLu,
            //integer overflow possibility here
            pooling_window_size: 3,
            pooling_stride: 2,
            pooling_type: convolutional_neural_network::PoolingType::MaxPooling,
        };

    let model_cnn: Vec<convolutional_neural_network::ConvolutionalBlock> =
        convolutional_neural_network::init_cnn(
            &Vec::new(),
            &cnn_model_settings.number_of_conv_blocks,
            &cnn_model_settings.number_of_kernels,
            &cnn_model_settings.kernel_size,
            &cnn_model_settings.kernel_stride,
            &cnn_model_settings.activation_type,
            &cnn_model_settings.pooling_window_size,
            &cnn_model_settings.pooling_stride,
            &cnn_model_settings.pooling_type,
        );

    // number_dataset::normalize(&mut training_set.0);
    // number_dataset::normalize(&mut testing_set.0);
    let slice: usize = 36;
    let (training_cost_history, dense_network): (Vec<f64>, Vec<ml::neural_network::Layer>) =
        convolutional_neural_network::train_model_cnn(
            &training_set.0[0..slice].to_vec(),
            &training_set.1[0..slice].to_vec(),
            &mut (model_cnn.clone(), None),
            &model_settings,
            &ml::neural_network::ActivationFunction::Softmax,
            &Some(convolutional_neural_network::LossFunctionCnn::CrossEntropyLoss),
            &ml::AvailableDatasets::Numbers,
        );

    let accuracy: f64 = convolutional_neural_network::test_model(
        //fix the clones, it increases runtime duration
        &mut (model_cnn.clone(), dense_network.clone()),
        &(
            testing_set.0[0..slice].to_vec(),
            testing_set.1[0..slice].to_vec(),
        ),
    );
    // dbg!(thing);
    ((model_cnn, dense_network), accuracy)
}
