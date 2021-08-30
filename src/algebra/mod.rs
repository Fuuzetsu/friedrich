//! Operations on matrix
//!
//! Various internal operations

mod extendable_matrix;

pub use extendable_matrix::{EMatrix, EVector};

use crate::parameters::kernel::Kernel;
use nalgebra::{
    storage::Storage, Cholesky, ComplexField, DMatrix, DVector, Dynamic, Matrix, SliceStorage, U1,
};

//-----------------------------------------------------------------------------
// ARBITRARY STORAGE TYPES

/// matrix with arbitrary storage
/// S: Storage<f64, Dynamic, Dynamic>
pub type SMatrix<S> = Matrix<f64, Dynamic, Dynamic, S>;

/// row vector with arbitrary storage
/// S: Storage<f64, U1, Dynamic>
pub type SRowVector<S> = Matrix<f64, U1, Dynamic, S>;

/// vector with arbitrary storage
/// S: Storage<f64, Dynamic, U1>
pub type SVector<S> = Matrix<f64, Dynamic, U1, S>;

//-----------------------------------------------------------------------------
// SLICE TYPES

/// represents a slice of a matrix
pub type MatrixSlice<'a> =
    Matrix<f64, Dynamic, Dynamic, SliceStorage<'a, f64, Dynamic, Dynamic, U1, Dynamic>>;

/// represents a view to a column from a matrix
pub type VectorSlice<'a> =
    Matrix<f64, Dynamic, U1, SliceStorage<'a, f64, Dynamic, U1, U1, Dynamic>>;

//-----------------------------------------------------------------------------
// COVARIANCE MATRIX

/// computes a covariance matrix using a given kernel and two matrices
/// the output has one row per row in m1 and one column per row in m2
pub fn make_covariance_matrix<
    S1: Storage<f64, Dynamic, Dynamic>,
    S2: Storage<f64, Dynamic, Dynamic>,
    K: Kernel,
>(
    m1: &SMatrix<S1>,
    m2: &SMatrix<S2>,
    kernel: &K,
) -> DMatrix<f64> {
    DMatrix::<f64>::from_fn(m1.nrows(), m2.nrows(), |r, c| {
        let x = m1.row(r);
        let y = m2.row(c);
        kernel.kernel(&x, &y)
    })
}

/// computes the cholesky decomposition of the covariance matrix of some inputs
/// adds a given diagonal noise
/// relies on the fact that only the lower triangular part of the matrix is needed for the decomposition
pub fn make_cholesky_cov_matrix<S: Storage<f64, Dynamic, Dynamic>, K: Kernel>(
    inputs: &SMatrix<S>,
    kernel: &K,
    diagonal_noise: f64,
    cholesky_epsilon: Option<f64>,
) -> Cholesky<f64, Dynamic> {
    // let make_covmatrix = || {

    // };

    // empty covariance matrix
    // TODO it would be faster to start with an an uninitialized matrix but it would require unsafe
    let mut covmatix = DMatrix::<f64>::from_element(inputs.nrows(), inputs.nrows(), std::f64::NAN);

    // computes the covariance for all the lower triangular matrix
    for (col_index, x) in inputs.row_iter().enumerate() {
        for (row_index, y) in inputs.row_iter().enumerate().skip(col_index) {
            covmatix[(row_index, col_index)] = kernel.kernel(&x, &y);
        }

        // adds diagonal noise
        covmatix[(col_index, col_index)] += diagonal_noise * diagonal_noise;
    }

    if let Some(cholesky_epsilon) = cholesky_epsilon {
        // If we fail even with explicit epsilon, report it back. The user may
        // have set it too low.
        cholesky_with_epsilon(covmatix, cholesky_epsilon).unwrap_or_else(|| {
            panic!(
                "Cholesky decomposition failed, even with epsilon of {}",
                cholesky_epsilon,
            )
        })
    } else {
        covmatix.cholesky().expect("Cholesky decomposition failed!")
    }
}

/// Similar to
/// https://docs.rs/nalgebra/0.29.0/src/nalgebra/linalg/cholesky.rs.html#166-201
/// but where it would normally fail, the given epsilon value is inserted
/// instead.
fn cholesky_with_epsilon(mut matrix: DMatrix<f64>, epsilon: f64) -> Option<Cholesky<f64, Dynamic>> {
    assert!(matrix.is_square(), "The input matrix must be square.");

    let n = matrix.nrows();

    for j in 0..n {
        for k in 0..j {
            let factor = unsafe { -matrix.get_unchecked((j, k)) };

            let (mut col_j, col_k) = matrix.columns_range_pair_mut(j, k);
            let mut col_j = col_j.rows_range_mut(j..);
            let col_k = col_k.rows_range(j..);

            col_j.axpy(factor.conjugate(), &col_k, 1.0);
        }

        let denom = unsafe { matrix.get_unchecked((j, j)) }.try_sqrt();
        // If we end up with a value that's 0 (or very close) or negative or a
        // NaN (maybe the original value was very tiny) then replace the value
        // with user epsilon instead.
        let denom = match denom {
            // Ensure that even if we have sqrt, it's not 0 (or extremely
            // close).
            Some(denom) if denom > f64::EPSILON => denom,
            _ => epsilon,
        };

        unsafe {
            *matrix.get_unchecked_mut((j, j)) = denom;
        }

        let mut col = matrix.slice_range_mut(j + 1.., j);
        col /= denom;
    }

    Some(Cholesky { chol: matrix })
}

/// add rows to the covariance matrix by updating its Cholesky decomposition in place
/// this is a O(n²*c) operation where n is the number of rows of the covariance matrix and c the number of new rows
/// `all_inputs` is a matrix with one row per input, the `nb_new_inputs` last rows are the one we want to add
pub fn add_rows_cholesky_cov_matrix<S: Storage<f64, Dynamic, Dynamic>, K: Kernel>(
    covmat_cholesky: &mut Cholesky<f64, Dynamic>,
    all_inputs: &SMatrix<S>,
    nb_new_inputs: usize,
    kernel: &K,
    diagonal_noise: f64,
) {
    // extracts the number of old inputs and new inputs from full inputs
    let nb_old_inputs = all_inputs.nrows() - nb_new_inputs;
    let new_inputs = all_inputs.rows(nb_old_inputs, nb_new_inputs);

    // add samples one row at a time
    for (row_index, row) in new_inputs.row_iter().enumerate() {
        // index where the column will be added in the Cholesky decomposition
        let col_index = nb_old_inputs + row_index;

        // computes the column, the covariance between the new row and previous rows
        let column_size = col_index + 1;
        let mut new_column = DVector::<f64>::from_fn(column_size, |training_row_index, _| {
            let training_row = all_inputs.row(training_row_index);
            kernel.kernel(&training_row, &row)
        });

        // add diagonal noise
        new_column[col_index] += diagonal_noise * diagonal_noise;

        // updates the cholesky decomposition with O(n²) operation
        *covmat_cholesky = covmat_cholesky.insert_column(col_index, new_column);
    }
}

/// Returns a vector with the gradient of the covariance matrix (which is a matrix) for each kernel parameter.
pub fn make_gradient_covariance_matrices<S: Storage<f64, Dynamic, Dynamic>, K: Kernel>(
    inputs: &SMatrix<S>,
    kernel: &K,
) -> Vec<DMatrix<f64>> {
    // empty covariance matrices
    let mut covmatrices: Vec<_> = (0..kernel.nb_parameters())
        .map(|_| DMatrix::<f64>::from_element(inputs.nrows(), inputs.nrows(), std::f64::NAN))
        .collect();

    // computes the covariance for all the lower triangular matrix
    for (col_index, x) in inputs.row_iter().enumerate() {
        for (row_index, y) in inputs.row_iter().enumerate().skip(col_index) {
            for (&grad, mat) in kernel.gradient(&x, &y).iter().zip(covmatrices.iter_mut()) {
                mat[(row_index, col_index)] = grad;
                mat[(col_index, row_index)] = grad;
            }
        }
    }

    covmatrices
}
