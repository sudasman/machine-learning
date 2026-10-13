use machine_learning::housing_price_dataset::generate_training_dataset as generate_housing;
use machine_learning::ml::{self, Layer};
use machine_learning::numpy::functions as function;
use machine_learning::numpy::matrix_arithmetic as array;

//to make it cleaner, make a struct named model_settings
//the fields will contain the data delcared by the let statements
//ORGANIZEEEE PLEASE
fn main() {
    let number_of_datasets = 1000;
    let data: (Vec<Vec<f64>>, Vec<Vec<f64>>) = generate_housing(number_of_datasets);
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

    //0.0 -> training x split, 0.1 -> training y split
    //1.0 -> testing x split, 1.1 -> testing y split
    let mut split_dataset: (
        (Vec<Vec<f64>>, Vec<Vec<f64>>),
        (Vec<Vec<f64>>, Vec<Vec<f64>>),
    ) = ml::train_test_split(&data.0, &data.1, 0.8);

    split_dataset.1.0 = machine_learning::housing_price_dataset::standardize_data(
        &split_dataset.0.0,
        Some(&mut split_dataset.1.0),
    );
    split_dataset.0.0 = machine_learning::housing_price_dataset::standardize_data(&split_dataset.0.0, None);
    
    ml::eval_model_training(&ml::train_model(
        &split_dataset.0,
        &mut model,
        learning_rate,
        epochs,
        batch_size,
    ));

    ml::test_model(&mut model, &split_dataset.1, 219000.0, 935000.0, 15000.0);
}

//JUNCTION LAST YEAR BUSINESS CASE EXAM
