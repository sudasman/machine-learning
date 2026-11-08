use machine_learning::housing_price_predictor;
use machine_learning::number_letter_recognition;
use machine_learning::number_recognition;
mod save;

//cargo run --release --features train
//caffeinate -i cargo run --release
#[cfg(feature = "train")]
fn main() {
    //train and save the model if better performance (experimental)

    // machine_learning::ml::train_and_save_cnn_model(
    //     machine_learning::ml::AvailableDatasets::Numbers,
    // );

    machine_learning::ml::train_and_save_cnn_model(
        machine_learning::ml::AvailableDatasets::NumbersLetters,
    );
}

#[cfg(not(feature = "train"))]
fn main() {
    number_letter_recognition::sample().expect("Error");
}
