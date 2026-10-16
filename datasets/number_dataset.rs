use std::fs;

pub fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

//Vec<Vec<Vec<(f64, f64, f64)>>> for rgb
pub fn load_images(path: &str) -> Vec<Vec<Vec<f64>>> {
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

pub fn load_dataset() {
    let images = load_images("datasets/mnist/train-images.idx3-ubyte");
    // dbg!(images);

    for row in &images[0] {
        let line: String = row.iter().map(|&p| if p > 0.5 { '#' } else { '.' }).collect();
        println!("{}", line);
    }
}
