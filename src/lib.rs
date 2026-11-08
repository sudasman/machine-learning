#[path = "../datasets/emnist.rs"]
mod emnist;
#[path = "../datasets/housing_price_dataset.rs"]
mod housing_price_dataset;
mod macros;
pub mod ml;
#[path = "../datasets/number_dataset.rs"]
mod number_dataset;
mod numpy;
mod save;
#[path = "../models/housing_price_predictor.rs"]
pub mod housing_price_predictor;
#[path = "../models/number_letter_recognition.rs"]
pub mod number_letter_recognition;
#[path = "../models/number_recognition.rs"]
pub mod number_recognition;
