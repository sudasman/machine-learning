use image::imageops::FilterType;

//will be in rgb
//channel corresponds to the rgb if channel == 1 then its grey scale
//[image][channel][row][col]
pub fn load_dataset(
    directory: &str,
    slice: usize,
) -> Result<Vec<Vec<Vec<Vec<f64>>>>, image::ImageError> {
    let mut images: Vec<Vec<Vec<Vec<f64>>>> = Vec::new();
    //the images are 512x512
    //could also use gaussian, but it takes a while to load

    for entry in std::fs::read_dir(directory)? {
        //could probably be more idiomatic
        if images.len() == slice {
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

pub fn generate_label() {}
