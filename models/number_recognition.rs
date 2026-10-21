use crate::number_dataset;
pub fn new() {
    //training_set = (training_images, training_labels)
    //testing_set = (testing_images, testing_labels)
    let (training_set, testing_set): (
        (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
        (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
    ) = number_dataset::load_dataset();

    dbg!(&training_set.1);
    // number_dataset::show_sample_image(&training_set.0, &training_set.1, 0, Some(7));
}
