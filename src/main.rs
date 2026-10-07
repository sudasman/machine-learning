#[path = "../datasets/housing_price_dataset.rs"]
mod housing_price_dataset;
mod ml;
mod numpy;
use crate::housing_price_dataset::{
    generate_training_dataset as generate_housing, unscale_price as price,
};
use ml::Layer as Layer;
use numpy::functions as function;
use numpy::matrix_arithmetic as array;

fn main() {
    let number_of_datasets = 5;
    let data: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) = generate_housing(number_of_datasets);
    let number_of_neurons_hidden_layer = vec![8];
    let number_of_hidden_layers = 1;
    let input_layer: Layer = ml::init_input_layer(&data.0[0]);
    let model: Vec<Layer> = ml::init_model(
        &number_of_neurons_hidden_layer,
        input_layer,
        &number_of_hidden_layers,
    );

    dbg!(ml::train_model(
        data.0,
        model,
        &number_of_neurons_hidden_layer,
        &number_of_hidden_layers
    ));
}
