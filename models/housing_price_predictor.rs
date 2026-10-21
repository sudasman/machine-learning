use crate::housing_price_dataset::{self, generate_training_dataset as generate_housing};
use crate::ml::neural_network::{self, AvailableDatasets, Dataset, Layer, ModelSettings};
use crate::numpy::functions as function;
use crate::numpy::matrix_arithmetic as array;

pub fn new() {
    let model_settings: ModelSettings = neural_network::ModelSettings {
        number_of_neurons_hidden_layer: vec![5],
        number_of_hidden_layers: 1,
        learning_rate: 0.01,
        batch_size: 36,
        epochs: 10000,
    };

    let number_of_datasets = 1000;
    //number_of_datasets is consumed here
    let mut data: Dataset = neural_network::Dataset::new(number_of_datasets);
    data.generate_dataset(AvailableDatasets::Housing);

    let mut model: Vec<Layer> = neural_network::init_model(
        &model_settings.number_of_neurons_hidden_layer,
        neural_network::init_input_layer(&data.dataset.0),
        &model_settings.number_of_hidden_layers,
    );

    //0.0 -> training x split, 0.1 -> training y split
    //1.0 -> testing x split, 1.1 -> testing y split
    let ((mut train_x, train_y), (mut test_x, test_y)): (
        (Vec<Vec<f64>>, Vec<Vec<f64>>),
        (Vec<Vec<f64>>, Vec<Vec<f64>>),
    ) = neural_network::train_test_split(&data.dataset.0, &data.dataset.1, 0.8);

    //standardize the x training and x testing splits
    test_x = housing_price_dataset::standardize_data(&train_x, Some(&mut test_x));
    train_x = housing_price_dataset::standardize_data(&train_x, None);

    let split_dataset: (
        (Vec<Vec<f64>>, Vec<Vec<f64>>),
        (Vec<Vec<f64>>, Vec<Vec<f64>>),
    ) = ((train_x, train_y), (test_x, test_y));
    neural_network::eval_model_training(&neural_network::train_model(
        &split_dataset.0,
        &mut model,
        model_settings.learning_rate,
        model_settings.epochs,
        model_settings.batch_size,
        Some(neural_network::LossFunction::Mse),
        None,
    ));

    neural_network::test_model(&mut model, &split_dataset.1, 219000.0, 935000.0, 15000.0);
}
