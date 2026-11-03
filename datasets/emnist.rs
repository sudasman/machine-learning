use std::fs::File;
use std::io::{BufRead, BufReader};

//csv structure -> row: label, image(grey scaled pixels)
pub fn load_csv(path: &str) -> std::io::Result<(Vec<u8>, Vec<Vec<Vec<f64>>>)> {
    //buffer the read
    let reader = BufReader::new(File::open(path)?);
    let mut labels: Vec<u8> = Vec::new();
    let mut images: Vec<Vec<Vec<f64>>> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut temp = line.split(",");
        labels.push(temp.next().unwrap().parse::<u8>().unwrap());
        let mut temp2 = temp
            .map(|val| val.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();
        //28x28 images
        let mut pixels = temp2
            .chunks(28)
            .map(|row| {
                row.iter()
                    .map(|val| *val as f64 / 255.0)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        images.push(pixels);
    }

    //labels store the index that corresponds to its respective ASCII value (mapping.txt)
    Ok((labels, images))
}

pub fn display_sample((labels, images): (&[u8], &[Vec<Vec<f64>>]), index: usize) {
    const ASCII: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    //without as char it would be the literal ascii value
    let label = ASCII[labels[index] as usize] as char;

    //could also hardcode and do 0..28
    for rows in images[index].iter() {
        let line = rows
            .iter()
            //|&pixel| (reference destructuring)
            .map(|pixel| if *pixel > 0.5 { "*" } else { " " })
            .collect::<String>();
        println!("{}", line);
    }
    println!("Label: {}", label);
}
