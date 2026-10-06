//creates a nxm matrix
pub fn create_matrix(n :usize, m: usize) -> Vec<Vec<f64>>{
    vec![vec![0.0; m]; n]
}

pub fn matrix_multiplication(matrix1: Vec<Vec<f64>>, matrix2: Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    //linear algebra 101 
    assert_eq!(matrix1[0].len(), matrix2.len());

    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; matrix2[0].len()]; matrix1.len()];
    for i in 0..matrix1.len(){
        for j in 0..matrix2[0].len(){
            for k in 0..matrix2.len(){
                matrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    matrix
}

pub fn matrix_addition(matrix1: Vec<Vec<f64>>, matrix2: Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    assert_eq!((matrix1.len(), matrix1[0].len()), (matrix2.len(), matrix2[0].len()));

    //either matrix1 or matrix2 works here
    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; matrix1[0].len()]; matrix1.len()];
    for i in 0..matrix1.len(){
        for j in 0..matrix1[0].len(){
            matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    matrix
}
