use crate::cancer_recognition::{Class, Subclass};
use image::imageops::FilterType;
//will be in rgb
//channel corresponds to the rgb if channel == 1 then its grey scale
//[image][channel][row][col]
pub fn load_dataset(
    directory: &str,
    slice: &usize,
) -> Result<Vec<Vec<Vec<Vec<f64>>>>, image::ImageError> {
    let mut images: Vec<Vec<Vec<Vec<f64>>>> = Vec::new();
    //the images are 512x512
    //could also use gaussian, but it takes a while to load

    for entry in std::fs::read_dir(directory)? {
        //could probably be more idiomatic
        if images.len() == *slice {
            break;
        }
        let path = entry?.path();

        let img = image::open(&path)?
            .resize_exact(512, 512, FilterType::Triangle)
            .to_rgb8();

        let (width, height) = (img.width() as usize, img.height() as usize);

        //[3][512][512]
        let mut pixels = vec![vec![vec![0.0; height as usize]; width as usize]; 3];

        for (col, row, pixel) in img.enumerate_pixels() {
            pixels[0][row as usize][col as usize] = pixel[0] as f64 / 255.0;
            pixels[1][row as usize][col as usize] = pixel[1] as f64 / 255.0;
            pixels[2][row as usize][col as usize] = pixel[2] as f64 / 255.0;
        }
        images.push(pixels);
    }
    Ok(images)
}

//.0 == class .1 == subclass

//mapping

//class
// 0 = Acute Lymphoblastic Leukemia (ALL)
// 1 = Brain Cancer
// 2 = Breast Cancer
// 3 = Cervical Cancer
// 4 = Kidney Cancer
// 5 = Lung and Colon Cancer
// 6 = Lymphoma
// 7 = Oral Cancer

//made f64 for the sake of not breaking existing implemntations
//everything will be shuffled in the end to mimic a real dataset
pub fn generate_labels(
    slice: &usize,
    class: Class,
    subclass: Subclass,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    (
        match class {
            Class::AcuteLymphoblasticLeukemia => vec![vec![0.0]; *slice],
            Class::BrainCancer => vec![vec![1.0]; *slice],
            Class::BreastCancer => vec![vec![2.0]; *slice],
            Class::CervicalCancer => vec![vec![3.0]; *slice],
            Class::KidneyCancer => vec![vec![4.0]; *slice],
            Class::LungCancer => vec![vec![5.0]; *slice],
            Class::ColonCancer => vec![vec![6.0]; *slice],
            Class::Lymphoma => vec![vec![7.0]; *slice],
            Class::OralCancer => vec![vec![8.0]; *slice],
        },
        match subclass {
            Subclass::AcuteLymphoblasticLeukemiaBenign => vec![vec![0.0]; *slice],
            Subclass::AcuteLymphoblasticLeukemiaEarly => vec![vec![1.0]; *slice],
            Subclass::AcuteLymphoblasticLeukemiaPre => vec![vec![2.0]; *slice],
            Subclass::AcuteLymphoblasticLeukemiaPro => vec![vec![3.0]; *slice],

            Subclass::BrainGlioma => vec![vec![4.0]; *slice],
            Subclass::BrainMeningioma => vec![vec![5.0]; *slice],
            Subclass::BrainTumor => vec![vec![6.0]; *slice],

            Subclass::BreastBenign => vec![vec![7.0]; *slice],
            Subclass::BreastMalignant => vec![vec![8.0]; *slice],

            Subclass::CervicalDyskeratotic => vec![vec![9.0]; *slice],
            Subclass::CervicalKoilocytotic => vec![vec![10.0]; *slice],
            Subclass::CervicalMetaplastic => vec![vec![11.0]; *slice],
            Subclass::CervicalParabasal => vec![vec![12.0]; *slice],
            Subclass::CervicalSuperficialIntermediate => vec![vec![13.0]; *slice],

            Subclass::KidneyNormal => vec![vec![14.0]; *slice],
            Subclass::KidneyTumor => vec![vec![15.0]; *slice],

            Subclass::LungAdenocarcinoma => vec![vec![16.0]; *slice],
            Subclass::LungBenign => vec![vec![17.0]; *slice],
            Subclass::LungSquamousCellCarcinoma => vec![vec![18.0]; *slice],

            Subclass::ColonAdenocarcinoma => vec![vec![19.0]; *slice],
            Subclass::ColonBenign => vec![vec![20.0]; *slice],

            Subclass::LymphomaChronicLymphocyticLeukemia => vec![vec![21.0]; *slice],
            Subclass::LymphomaFollicularLymphoma => vec![vec![22.0]; *slice],
            Subclass::LymphomaMantleCellLymphoma => vec![vec![23.0]; *slice],

            Subclass::OralNormal => vec![vec![24.0]; *slice],
            Subclass::OralOralSquamousCellCarcinoma => vec![vec![25.0]; *slice],
        },
    )
}
