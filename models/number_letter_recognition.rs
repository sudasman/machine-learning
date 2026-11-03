use crate::emnist;
use crate::save;

#[derive(serde::Serialize, serde::Deserialize)]
struct Dataset {
    labels: Vec<u8>,
    images: Vec<Vec<Vec<f64>>>,
}

pub fn new() {
    if let Ok(dataset) = save::load_cache::<Dataset>("datasets/emnist/cache/emnist_train.bin") {
        println!("Retrieving cache");
        let (labels, images): (Vec<u8>, Vec<Vec<Vec<f64>>>) = (dataset.labels, dataset.images);
    } else {
        let (labels, images): (Vec<u8>, Vec<Vec<Vec<f64>>>) =
            emnist::load_csv("datasets/emnist/emnist-byclass-train.csv").expect("invalid path");
        save::cache(&Dataset { labels, images }, "datasets/emnist/cache/emnist_train.bin").expect("Failed to create cache");;
        println!("cached");
    }

    // emnist::display_sample((&labels, &images), 1);
}
