use machine_learning::housing_price_predictor;
use machine_learning::number_recognition;
use serde_json;
mod save;

type Model = (
    Vec<machine_learning::ml::convolutional_neural_network::ConvolutionalBlock>,
    Vec<machine_learning::ml::neural_network::Layer>,
);

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SavedModel {
    pub model: Model,
    pub accuracy: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //housing_price_predictor::new();
    let ((model_cnn, dense_network), accuracy): (
        (
            Vec<machine_learning::ml::convolutional_neural_network::ConvolutionalBlock>,
            Vec<machine_learning::ml::neural_network::Layer>,
        ),
        f64,
    ) = number_recognition::new();

    let saved_model: SavedModel = SavedModel {
        model: (model_cnn, dense_network),
        accuracy,
    };
    //todo: save the housing price model too
    save::save_model(
        &saved_model,
        "models/number_recognition_model_parameters/cnn_model.json",
    )?;
    Ok(())
}

