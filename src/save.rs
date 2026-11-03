use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub fn save_model<T: serde::Serialize>(
    model: &T,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let writer = BufWriter::new(File::create(path)?);
    //pretty print
    serde_json::to_writer_pretty(writer, model)?;
    Ok(())
}

pub fn load_model<T>(path: &str) -> Result<T, Box<dyn std::error::Error>>
where
    //could also use deserialzeowned
    T: for<'a> serde::Deserialize<'a>,
{
    let reader = BufReader::new(File::open(path)?);
    Ok(serde_json::from_reader(reader)?)
}

pub fn cache<T: serde::Serialize>(temp: &T, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let writer = BufWriter::new(File::create(path)?);

    bincode::serialize_into(writer, temp)?;

    Ok(())
}

pub fn load_cache<T>(path: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: for<'a> serde::Deserialize<'a>,
{
    let reader = BufReader::new(File::open(path)?);

    let data = bincode::deserialize_from(reader)?;

    Ok(data)
}
