#[path = "../datasets/housing_price_dataset.rs"]
mod housing_price_dataset;
mod ml;
mod numpy;
use crate::housing_price_dataset::generate_training_dataset as generate_housing;
use ml::Layer;
use numpy::functions as function;
use numpy::matrix_arithmetic as array;

fn main() {
    let number_of_datasets = 30;
    let (training, actual): (Vec<Vec<f64>>, Vec<Vec<f64>>) = generate_housing(number_of_datasets);
    let data: (Vec<Vec<f64>>, Vec<Vec<f64>>) = (crate::housing_price_dataset::standardize_data(&training), actual);
    let number_of_neurons_hidden_layer = vec![8];
    let number_of_hidden_layers = 1;
    let input_layer: Layer = ml::init_input_layer(&data.0);
    let mut model: Vec<Layer> = ml::init_model(
        &number_of_neurons_hidden_layer,
        input_layer,
        &number_of_hidden_layers,
    );

    let learning_rate: f64 = 0.001;
    let epochs: usize = 10000;
    //batch size must be less than or equal to number_of_datasets
    let batch_size: usize = 5;

    ml::eval_model_training(&ml::train_model(&data, &mut model, learning_rate, epochs, batch_size));
}

//JUNCTION LAST YEAR BUSINESS CASE EXAM
