//! Nalgebra based Gaussian process
//!
//! Used for raw acess to the underlying implementation.
//! Mostly usefull to reduce copies and provide binding for other input/output types.
//!
//! TODO illustrate usage

use crate::parameters::{kernel, kernel::Kernel, prior, prior::Prior};
use nalgebra::{Cholesky, Dynamic, DMatrix, DVector};
use crate::algebra::{EMatrix, EVector, make_cholesky_covariance_matrix, make_covariance_matrix};
use crate::conversion::InputMatrix;

mod multivariate_normal;
pub use multivariate_normal::MultivariateNormal;

mod builder;
pub use builder::GaussianProcessBuilder;

/// A Gaussian process that can be used to make predictions based on its training data
pub struct GaussianProcess<KernelType: Kernel, PriorType: Prior>
{
   /// value to which the process will regress in the absence of informations
   pub prior: PriorType,
   /// kernel used to fit the process on the data
   pub kernel: KernelType,
   /// amplitude of the noise of the data
   pub noise: f64,
   /// data used for fit
   training_inputs: EMatrix,
   training_outputs: EVector,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covmat_cholesky: Cholesky<f64, Dynamic>
}

impl GaussianProcess<kernel::Gaussian, prior::Constant>
{
   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn default<Input: InputMatrix>(training_inputs: Input, training_outputs: Input::InVector) -> Self
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
      .fit_kernel()
      .fit_prior()
      .train()
   }

   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn builder<Input: InputMatrix>(training_inputs: Input,
                                      training_outputs: Input::InVector)
                                      -> GaussianProcessBuilder<kernel::Gaussian, prior::Constant>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
   }
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType>
{
   /// builds a new gaussian process with the given inputs
   pub fn new<Input: InputMatrix>(prior: PriorType,
                                  kernel: KernelType,
                                  noise: f64,
                                  training_inputs: Input,
                                  training_outputs: Input::InVector)
                                  -> Self
   {
      let training_inputs = Input::into_dmatrix(training_inputs);
      let training_outputs = Input::into_dvector(training_outputs);
      assert_eq!(training_inputs.nrows(), training_outputs.nrows());
      // converts training data into extendable matrix
      let training_inputs = EMatrix::new(training_inputs);
      let training_outputs = EVector::new(training_outputs - prior.prior(&training_inputs.as_matrix()));
      // computes cholesky decomposition
      let covmat_cholesky = make_cholesky_covariance_matrix(&training_inputs.as_matrix(), &kernel, noise);
      GaussianProcess { prior, kernel, noise, training_inputs, training_outputs, covmat_cholesky }
   }

   //----------------------------------------------------------------------------------------------
   // FIT

   /// adds new samples to the model
   /// update the model (which is faster than a training from scratch)
   /// does not refit the parameters
   pub fn add_samples<Input: InputMatrix>(&mut self, inputs: &Input, outputs: &Input::InVector)
   {
      let inputs = Input::to_dmatrix(inputs);
      let outputs = Input::to_dvector(outputs);
      assert_eq!(inputs.nrows(), outputs.nrows());
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());
      // grows the training matrix
      let outputs = outputs - self.prior.prior(&inputs);
      self.training_inputs.add_rows(&inputs);
      self.training_outputs.add_rows(&outputs);
      // recompute cholesky matrix
      self.covmat_cholesky =
         make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
      // TODO update cholesky matrix instead of recomputing it from scratch
   }

   /// fits the parameters if requested and retrain the model from scratch if needed
   pub fn fit_parameters(&mut self, fit_prior: bool, fit_kernel: bool)
   {
      if fit_prior
      {
         // gets the original data back in order to update the prior
         let training_outputs =
            self.training_outputs.as_vector() + self.prior.prior(&self.training_inputs.as_matrix());
         self.prior.fit(&self.training_inputs.as_matrix(), &training_outputs);
         let training_outputs = training_outputs - self.prior.prior(&self.training_inputs.as_matrix());
         self.training_outputs.assign(&training_outputs);
         // NOTE: adding and substracting each time we fit a prior might be numerically unstable
      }

      if fit_kernel
      {
         // fit kernel using new data and new prior
         self.kernel.fit(&self.training_inputs.as_matrix(), &self.training_outputs.as_vector());
      }

      if fit_prior || fit_kernel
      {
         // retranis model if a fit happened
         self.covmat_cholesky =
            make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
      }
   }

   /// adds new samples to the model and fit the parameters
   /// faster than doing add_samples().fit_parameters()
   pub fn add_samples_fit<Input: InputMatrix>(&mut self,
                                              inputs: &Input,
                                              outputs: &Input::InVector,
                                              fit_prior: bool,
                                              fit_kernel: bool)
   {
      let inputs = Input::to_dmatrix(inputs);
      let outputs = Input::to_dvector(outputs);
      assert_eq!(inputs.nrows(), outputs.nrows());
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());
      // grows the training matrix
      let outputs = outputs - self.prior.prior(&inputs);
      self.training_inputs.add_rows(&inputs);
      self.training_outputs.add_rows(&outputs);
      // refit the parameters and retrain the model from scratch
      if fit_kernel || fit_prior
      {
         self.fit_parameters(fit_prior, fit_kernel);
      }
      else
      {
         // retrains the model anyway if no fit happened
         self.covmat_cholesky =
            make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
      }
   }

   //----------------------------------------------------------------------------------------------
   // PREDICT

   /// predicts the mean of the gaussian process at each row of the input
   pub fn predict<Input: InputMatrix>(&self, inputs: &Input) -> Input::OutVector
   {
      let inputs = Input::to_dmatrix(inputs);
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // computes weights to give each training sample
      let mut weights = make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      self.covmat_cholesky.solve_mut(&mut weights);

      // computes prior for the given inputs
      let mut prior = self.prior.prior(&inputs);

      // weights.transpose() * &self.training_outputs + prior
      prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);

      Input::from_dvector(&prior)
   }

   /// predicts the variance of the gaussian process at each row of the input
   pub fn predict_variance<Input: InputMatrix>(&self, inputs: &Input) -> Input::OutVector
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl).diagonal()
      // note that here the diagonal is just the sum of the squares of the values in the columns of kl
      let inputs = Input::to_dmatrix(inputs);
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // compute the weights
      let cov_train_inputs = make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // (cov_inputs_inputs - cov_train_inputs.transpose() * weights).diagonal()
      let mut variances = DVector::<f64>::zeros(inputs.nrows());
      for i in 0..inputs.nrows()
      {
         // Note that this might be done with a zipped iterator
         let input = inputs.row(i);
         let base_cov = self.kernel.kernel(&input, &input);
         let predicted_cov = cov_train_inputs.column(i).dot(&weights.column(i));
         variances[i] = base_cov - predicted_cov;
      }

      Input::from_dvector(&variances)
   }

   /// predicts the covariance of the gaussian process at each row of the input
   pub fn predict_covariance<Input: InputMatrix>(&self, inputs: &Input) -> DMatrix<f64>
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl)
      let inputs = Input::to_dmatrix(inputs);
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // compute the weights
      let cov_train_inputs = make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // computes the intra points covariance
      let mut cov_inputs_inputs = make_covariance_matrix(&inputs, &inputs, &self.kernel);

      // cov_inputs_inputs - cov_train_inputs.transpose() * weights
      cov_inputs_inputs.gemm_tr(-1f64, &cov_train_inputs, &weights, 1f64);
      cov_inputs_inputs
   }

   /// produces a structure that can be used to sample the gaussian process at the given points
   pub fn sample_at<Input: InputMatrix>(&self, inputs: &Input) -> MultivariateNormal<Input>
   {
      let inputs = Input::to_dmatrix(inputs);
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // compute the weights
      let cov_train_inputs = make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // computes covariance
      let mut cov_inputs_inputs = make_covariance_matrix(&inputs, &inputs, &self.kernel);
      cov_inputs_inputs.gemm_tr(-1f64, &cov_train_inputs, &weights, 1f64);
      let cov = cov_inputs_inputs;

      // computes the mean
      let mut prior = self.prior.prior(&inputs);
      prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);
      let mean = prior;

      MultivariateNormal::new(mean, cov)
   }
}
