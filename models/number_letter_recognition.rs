use crate::emnist;
use crate::save;

#[derive(serde::Serialize, serde::Deserialize)]
struct Dataset {
    labels: Vec<u8>,
    images: Vec<Vec<Vec<f64>>>,
}

pub fn new() {
    let dataset: Dataset = if let Ok(dataset) =
        save::load_cache::<Dataset>("datasets/emnist/cache/emnist_train.bin")
    {
        println!("Retrieving cache");
        dataset
    } else {
        let (labels, images): (Vec<u8>, Vec<Vec<Vec<f64>>>) =
            emnist::load_csv("datasets/emnist/emnist-byclass-train.csv").expect("invalid path");

        let dataset: Dataset = Dataset { labels, images };
        save::cache(&dataset, "datasets/emnist/cache/emnist_train.bin")
            .expect("Failed to create cache");
        println!("cached");
        dataset
    };

    emnist::display_sample((&dataset.labels, &dataset.images), 1);
}
