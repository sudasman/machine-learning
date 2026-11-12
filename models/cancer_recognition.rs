use crate::cancer;

pub fn new() {
    let images = cancer::load_dataset(
        "datasets/multi_cancer/Multi Cancer/Cervical Cancer/cervix_dyk",
        1,
    )
    .expect("error");
    dbg!(&images[0]);
}
