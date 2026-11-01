use machine_learning::housing_price_predictor;
use machine_learning::number_dataset;
use machine_learning::number_recognition;
mod save;

//cargo run --release --features train
#[cfg(feature = "train")]
fn main() {
    //train and save the model if better performance (experimental)
    machine_learning::ml::train_and_save_cnn_model(
        machine_learning::ml::AvailableDatasets::Numbers,
    );
}

#[cfg(not(feature = "train"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model: machine_learning::ml::SavedModel = save::load_model("models/number_recognition_model_parameters/cnn_model.json")?;

    let (image, label) = number_dataset::grab_random_image();

    number_dataset::show_sample_image(&vec![image.clone()], &vec![label.clone()], 0, None);
    let (_, prediction) = machine_learning::ml::convolutional_neural_network::test_model(
        &mut model.model.0,
        &mut model.model.1,
        &(vec![image.clone()], vec![label.clone()]),
    );

    println!("{:#?}", prediction);
    Ok(())
}
