use crate::ml::convolutional_neural_network;
use crate::number_dataset;

pub fn new() {
    //training_set = (training_images, training_labels)
    //testing_set = (testing_images, testing_labels)
    let (training_set, testing_set): (
        (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
        (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
    ) = number_dataset::load_dataset();

    // dbg!(&training_set.1);
    // number_dataset::show_sample_image(&training_set.0, &training_set.1, 0, Some(7));

    let cnn_model_settings: convolutional_neural_network::cnn_model_settings =
        convolutional_neural_network::cnn_model_settings {
            number_of_conv_blocks: 1,
            //index corresponds to each block
            number_of_kernels: vec![3],
            kernel_size: vec![3],
            kernel_stride: vec![2],
            activation_type: convolutional_neural_network::ActivationFunctionCnn::ReLu,
            pooling_window_size: 3,
            pooling_stride: 2,
            pooling_type: convolutional_neural_network::PoolingType::MaxPooling,
        };

    let thing: Vec<convolutional_neural_network::ConvolutionalBlock> =
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
        
    dbg!(thing);
}
