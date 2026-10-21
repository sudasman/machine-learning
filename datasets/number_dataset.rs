use std::fs;

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    //relative addressing
    //be -> big endian -> most significant byte first (reading left to right), little endian is the opposite
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

//Vec<Vec<Vec<(f64, f64, f64)>>> for rgb
fn load_images(path: &str) -> Vec<Vec<Vec<f64>>> {
    let bytes: Vec<u8> = fs::read(path).expect("path not found");
    //assume correct file structure
    //idx3 structure -> byte 0: magic number -> byte 4: number of images -> byte 8: rows -> byte 12 columns, bytes 0-16: header
    let number_of_images = read_u32(&bytes, 4) as usize;
    let rows = read_u32(&bytes, 8) as usize;
    let columns = read_u32(&bytes, 12) as usize;

    bytes[16..]
        //divide the dataset into individual images
        //each pixel is a byte (8 bits)
        //therefore individual images take up rows*columns bytes
        .chunks(rows * columns)
        //chunks until number_of_images
        .take(number_of_images)
        .map(|image| {
            image
                .chunks(columns)
                .map(|row| {
                    row.iter()
                        //brightness
                        .map(|&pixel| pixel as f64 / 255.0)
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>()
        })
        .collect::<Vec<Vec<Vec<f64>>>>()
}

//**maybe bug in the future, the label datatype is a float (originally u8)**
fn load_labels(path: &str) -> Vec<Vec<f64>> {
    let bytes = fs::read(path).expect("path not found");
    //remember: take(usize) stops the iterator at usize
    bytes[8..]
        .iter()
        .take(read_u32(&bytes, 4) as usize)
        .map(|&label| vec![label as f64])
        .collect::<Vec<Vec<f64>>>()
}

pub fn show_sample_image(
    images: &Vec<Vec<Vec<f64>>>,
    labels: &Vec<Vec<f64>>,
    single_n: usize,
    multiple_n: Option<usize>,
) {
    if let Some(n) = multiple_n {
        for i in 0..n {
            //images is a tensor
            //either &images[n] or explicitly using images[n].iter() works
            for rows in images[i].iter() {
                let line = rows
                    .iter()
                    .map(|&pixel| if pixel > 0.5 { "*" } else { " " })
                    .collect::<String>();
                println!("{}", line);
            }
            println!("Number: {}", labels[i][0]);
        }
    } else {
        for rows in images[single_n].iter() {
            let line = rows
                .iter()
                .map(|&pixel| if pixel > 0.5 { "*" } else { " " })
                .collect::<String>();
            println!("{}", line);
        }
        println!("Number: {}", labels[single_n][0]);
    }
}

//task for tomorrow: seperate into training and testing sets (return type be tuple)
pub fn load_dataset() -> (
    (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
    (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
) {
    //t10k -> testing set, idx3 stores tensors, idx1 -> 1D

    let training_images: Vec<Vec<Vec<f64>>> = load_images("datasets/mnist/train-images.idx3-ubyte");
    let training_labels: Vec<Vec<f64>> = load_labels("datasets/mnist/train-labels.idx1-ubyte");

    let testing_images: Vec<Vec<Vec<f64>>> = load_images("datasets/mnist/t10k-images.idx3-ubyte");
    let testing_labels: Vec<Vec<f64>> = load_labels("datasets/mnist/t10k-labels.idx1-ubyte");

    (
        (training_images, training_labels),
        (testing_images, testing_labels),
    )
}
