use image::imageops::FilterType;

//will be in rgb
pub fn load_dataset(directory: &str) -> Result<Vec<Vec<Vec<(f64, f64, f64)>>>, image::ImageError> {
    let mut images: Vec<Vec<Vec<(f64, f64, f64)>>> = Vec::new();
    //the images are 512x512
    //could also use gaussian, but it takes a while to load

    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();

        let img = image::open(&path)?
            .resize_exact(512, 512, FilterType::Triangle)
            .to_rgb8();

        let (width, height) = (img.width() as usize, img.height() as usize);
        //should be 512x512
        let mut pixels = vec![vec![(0.0, 0.0, 0.0); height]; width];

        for (row, col, pixel) in img.enumerate_pixels() {
            pixels[row as usize][col as usize] = (
                pixel[0] as f64 / 255.0,
                pixel[1] as f64 / 255.0,
                pixel[2] as f64 / 255.0,
            );
        }
        images.push(pixels);
    }
    Ok(images)
}
