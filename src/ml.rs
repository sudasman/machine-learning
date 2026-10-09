use crate::numpy::functions as function;
use crate::numpy::matrix_arithmetic as array;

#[derive(Debug)]
pub struct Layer {
    pub neurons: usize,
    pub input_matrix: Vec<Vec<f64>>,
    pub weight_matrix: Vec<Vec<f64>>,
    pub bias_matrix: Vec<Vec<f64>>,
    pub weighted_sum_matrix: Vec<Vec<f64>>,
}

impl Layer {
    pub fn new(input_matrix: Vec<Vec<f64>>, neurons: usize) -> Self {
        let weight_matrix: Vec<Vec<f64>> = array::create_matrix(input_matrix[0].len(), neurons, true);
        let bias_matrix: Vec<Vec<f64>> = array::create_matrix(1, neurons, true);
        let weighted_sum_matrix = weighted_sum_matrix(&input_matrix, &weight_matrix, &bias_matrix);

        Layer {
            neurons,
            input_matrix,
            weight_matrix,
            bias_matrix,
            weighted_sum_matrix,
        }
    }

    pub fn forward(&mut self, input_matrix: &Vec<Vec<f64>>,) {
        self.input_matrix = input_matrix.clone();
        self.weighted_sum_matrix = weighted_sum_matrix(&self.input_matrix, &self.weight_matrix, &self.bias_matrix);
    }
}

pub fn init_input_layer(training_data: &Vec<Vec<f64>>) -> Layer {
    Layer::new(training_data.clone(), training_data[0].len())
}

pub fn train_model(
    data: &(Vec<Vec<f64>>>, Vec<Vec<f64>>),
    model: &mut Vec<Layer>,
    learning_rate: f64,
    epoch: usize,
    batch_size: usize,
) -> Vec<f64> {
    assert!(batch_size > 0 && batch_size <= data.0.len());
    let mut cost_history: Vec<f64> = Vec::new();

    for epoch in 0..epochs {
        let (shuffled_training_dataset, shuffled_actual_value_dataset): (Vec<Vec<f64>>, Vec<Vec<f64>>) = function::shuffle_dataset(&data.0, &data.1);
        let (mut total, mut batches): (f64, usize) = (0.0, 0);

        for start in (0..shuffled_training_dataset.len()).step_by(batch_size) {
            let edge = (start + batch_size).min(shuffled_training_dataset.len());
            let training_batch = shuffled_training_dataset[start..end].to_vec();
            let actual_batch = shuffled_actual_value_dataset(start..end).to_vec();

            let prediction = forward_pass(model, &training_batch);

            total += function::MSE(&prediction, &actual_batch);
            backward_pass(model, &prediction, &actual_batch, learning_rate);
            batches+=1;
        }

        let avg: f64 = total / batchs as f64;
        cost_history.push(avg);

        if epoch % 100 == 0 {
            println!("Epoch: {} Cost: {}", epoch, avg);
        }
    }
    cost_history
}

pub fn init_model(
    number_of_neurons_hidden_layer: &Vec<usize>,
    input_layer: Layer,
    number_of_hidden_layers: &usize,
) -> Vec<Layer> {
    let mut model = init_hidden_layer(
        number_of_neurons_hidden_layer,
        input_layer,
        number_of_hidden_layers,
    );
    let output_layer: Layer = Layer::new(
        apply_activation_function(model[model.len() - 1].weighted_sum_matrix.clone()),
        1,
    );
    model.push(output_layer);
    model
}

pub fn init_hidden_layer(
    number_of_neurons_hidden_layer: &Vec<usize>,
    input_layer: Layer,
    number_of_hidden_layers: &usize,
) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();

    layers.push(input_layer);

    for i in 0..*number_of_hidden_layers {
        let hidden_layer: Layer = Layer::new(
            apply_activation_function(layers[layers.len() - 1].weighted_sum_matrix.clone()),
            number_of_neurons_hidden_layer[i],
        );

        layers.push(hidden_layer);
    }
    layers
}
pub fn weighted_sum_matrix(
    input_matrix: &Vec<Vec<f64>>,
    weight_matrix: &Vec<Vec<f64>>,
    bias_matrix: &Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    array::add_bias(
        &array::matrix_multiplication(input_matrix, weight_matrix),
        bias_matrix,
    )
}

pub fn apply_activation_function(weighted_sum_matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    weighted_sum_matrix.iter().map(|row|.iter().map(|value| function::sigmoid(value).collect())).collect()
}

pub fn forward_pass(model: &mut Vec<Layer>, batch: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut input = batch.clone();
    for (i, layer) in model.iter_mut().enumerate() {
        layer.forward(&input);
        //if we are at the last layer, the input matrix will be the weighted sum matrix
        input = if i == model.len() - 1 {
            layer.weighted_sum_matrix.clone()
        } else {
            //new input layer is just the weighted sum matrix after a activation fucntion of the previous layer 
            apply_activation_function(layer.weighted_sum_matrix.clone())
        };
        input
    }
}

fn backward_pass(model: &mut Vec<Layer>, prediction: &Vec<Vec<f64>>, actual: &Vec<Vec<f64>>, learning_rate: f64,) {
    function::stochastic_gradient_descent(model, prediction, actual, learning_rate);
}

