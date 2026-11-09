use crate::emnist;
use crate::ml::{self, convolutional_neural_network};
use crate::save;
#[derive(serde::Serialize, serde::Deserialize)]
struct Dataset {
    labels: Vec<Vec<f64>>,
    images: Vec<Vec<Vec<f64>>>,
}

impl Dataset {
    fn default() -> Self {
        Dataset {
            labels: Vec::new(),
            images: Vec::new(),
        }
    }
}

pub fn new() -> (
    (
        Vec<convolutional_neural_network::ConvolutionalBlock>,
        Vec<ml::neural_network::Layer>,
    ),
    f64,
) {
    let mut training_dataset: Dataset = Dataset::default();
    let mut testing_dataset: Dataset = Dataset::default();

    check_cache(
        &mut training_dataset,
        "datasets/emnist/cache/emnist_train.bin",
        "datasets/emnist/emnist-byclass-train.csv",
    );
    check_cache(
        &mut testing_dataset,
        "datasets/emnist/cache/emnist_testing.bin",
        "datasets/emnist/emnist-byclass-test.csv",
    );

    let model_settings: ml::neural_network::ModelSettings = ml::neural_network::ModelSettings {
        number_of_neurons_hidden_layer: vec![128, 64],
        number_of_hidden_layers: 2,
        learning_rate: 0.01,
        batch_size: 256,
        epochs: 100,
        optimization: ml::OptimizationAlgorithms::Adam,
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

    let mut model_cnn: Vec<convolutional_neural_network::ConvolutionalBlock> =
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

    //600k max
    let slice_training: usize = 300000;
    //100k max
    let slice_testing: usize = 45000;

    let training_set: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) =
        (training_dataset.images, training_dataset.labels);
    let testing_set: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) =
        (testing_dataset.images, testing_dataset.labels);

    let (training_cost_history, mut dense_network): (Vec<f64>, Vec<ml::neural_network::Layer>) =
        convolutional_neural_network::train_model_cnn(
            &training_set.0[0..slice_training].to_vec(),
            &training_set.1[0..slice_training].to_vec(),
            &mut model_cnn,
            None,
            &model_settings,
            &ml::neural_network::ActivationFunction::Softmax,
            &Some(convolutional_neural_network::LossFunctionCnn::CrossEntropyLoss),
            &ml::AvailableDatasets::NumbersLetters,
        );

    let (accuracy, _, _): (f64, Vec<Vec<f64>>, char) = convolutional_neural_network::test_model(
        &mut model_cnn,
        &mut dense_network,
        &(
            testing_set.0[0..slice_testing].to_vec(),
            testing_set.1[0..slice_testing].to_vec(),
        ),
        false,
    );

    ((model_cnn, dense_network), accuracy)

    // emnist::display_sample((&dataset.labels, &dataset.images), 1);
}

fn check_cache(mut dataset: &mut Dataset, path: &str, file: &str) {
    *dataset = if let Ok(dataset) = save::load_cache::<Dataset>(path) {
        println!("Retrieving cache");
        dataset
    } else {
        let (labels, images): (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) =
            emnist::load_csv(file).expect("invalid path");

        let dataset: Dataset = Dataset { labels, images };
        save::cache(&dataset, path).expect("Failed to create cache");
        println!("cached");
        dataset
    };
}

pub fn sample() -> Result<(Vec<String>, char), Box<dyn std::error::Error>> {
    let mut model = if let Ok(model) = save::load_model::<ml::SavedModel>(
        "/Users/eason/Desktop/rust/machine-learning/models/number_letter_recognition_model_parameters/cnn_model.json",
    ) {
        model
    } else {
        panic!("No model saved to sample");
    };
    let mut dataset: Dataset = Dataset::default();

    check_cache(
        &mut dataset,
        "/Users/eason/Desktop/rust/machine-learning/datasets/emnist/cache/emnist_testing.bin",
        "/Users/eason/Desktop/rust/machine-learning/datasets/emnist/emnist-byclass-test.csv",
    );

    let slice = 1000;
    let (label, image): (&[f64], &[Vec<f64>]) =
        emnist::grab_random_image(&dataset.images[..slice], &dataset.labels[..slice]);

    let temp = emnist::display_sample((&vec![label.to_vec()], &vec![image.to_vec()]));

    let (_, prediction, guess) = convolutional_neural_network::test_model(
        &mut model.model.0,
        &mut model.model.1,
        &(vec![image.to_vec()], vec![label.to_vec()]),
        true,
    );

    println!("Probability Distribution: {:#?}", prediction);

    //used for display for the web
    Ok((temp, guess))
}

//same functionality as sample but with a custom image
pub fn process_image() {}
