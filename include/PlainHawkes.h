/**
 * \file PlainHawkes.h
 * \brief The class definition of PlainHawkes implementing the standard Hawkes process.
 */
#ifndef PLAIN_HAWKES_H
#define PLAIN_HAWKES_H
#include <vector>
#include <string>
#include <map>
#include "Process.h"
#include "Optimizer.h"

/**
 * \class PlainHawkes PlainHawkes.h "include/PlainHawkes.h"
 * \brief PlainHawkes implements the standard multivariate Hawkes process.
 *
 * A multivariate Hawkes Process is a process where the occurrence of an event to a dimension will trigger more events on this dimension and other related dimensions in the near future. The intensity function of each dimension of the Hawkes process is defined as the following :
 * \f{align}{
 * 	\lambda^n(t) = \lambda_0^n + \sum_{m=1}^D\alpha_{mn}\sum_{t^m_j < t}\exp(-\beta_{mn}(t - t^m_j)),
 * \f}
 * where \f$\lambda_0^n\geq 0\f$ is the base intensity, \f$D\f$ is the number of dimensions, and \f$\alpha_{mn}\geq 0\f$ captures the extent to which an event on dimension m can trigger an event on dimension n in the near future. The collection of \f$\{\alpha_{mn}\}\f$ can be represented as a matrix \f$\mathbf{A}(m,n) = \alpha_{mn}\f$, and the collection of \f$\{\lambda_0^n\}\f$ can be represented as a column vector \f$\boldsymbol{\lambda}_0\f$.
 *
 * \f$\boldsymbol{\lambda}_0\f$ and \f$\mathbf{A}\f$ are the model parameters of a multivariate Hawkes process. The collection of \f$\{\beta_{mn}\geq 0\}\f$ is often given in advance. 
 */
class PlainHawkes : public IProcess
{

protected:

/**
 * \brief Beta_ is a D-by-D matrix, which stores the \f$\{\beta_{mn}\geq 0\}\f$ decaying rates of the exponential kernels. 
 */
	Eigen::MatrixXd Beta_;
  
/**
 * 	\brief all_exp_kernel_recursive_sum_ stores the cumulative influence between dimensions.
 * 	
 * 	For each sequence c, given a pair of dimension m and n, and a point \f$t^{c,n}_{i}\f$ on the dimension n, 
 * 	\f{align}{
 * 		\text{all_exp_kernel_recursive_sum_[c][m][n][i]} & = \sum_{t^{c,m}_{j} < t^{c,n}_{i}}\exp(-\beta_{mn}(t^{c,n}_i - t^{c,m}_j))\\
 * 		& = \exp(-\beta_{mn}(t^{c,n}_i - t^{c,n}_{i-1}))\cdot\text{all_exp_kernel_recursive_sum_[c][m][n][i - 1]}\quad + \sum_{t^{c,n}_{i-1}\leq t^{c,m}_j < t^{c,n}_i} \exp(-\beta_{mn}(t^{c,n}_i - t^{c,m}_j))
 * 	\f}
 * 	which is the sum of the influence from all past events \f$t^{c,m}_{j} < t^{c,n}_{i}\f$ on the dimension m in the sequence c.
 */
	std::vector<std::vector<std::vector<Eigen::VectorXd> > > all_exp_kernel_recursive_sum_;

/**
 *  \brief intensity_itegral_features_ stores the integral of each exponential kernel.
 *
 *	For each sequence c, given a pair of dimension m and n, 
 *	\f{align}{
 *	\text{intensity_itegral_features_[c](m,n)} = \sum_{t^{c,m}_j < T^c}(1 - \exp(-\beta_{mn}(T_c - t^{c,m}_j)))
 *	\f}
 */
	std::vector<Eigen::MatrixXd> intensity_itegral_features_;

/**
 * 	\brief observation_window_T_ stores the observation window for each sequence.
 *
 * 	For each sequence c, \f$\text{observation_window_T_}(c) = T_c\f$.
 */
	Eigen::VectorXd observation_window_T_;

/**
 * \brief RNG_ implements a simple random number generator.
 */
	SimpleRNG RNG_;

/**
 * 	\brief num_sequences_ is the total number of sequences in the given data.
 */
	unsigned num_sequences_;

/**
 * \brief Initialize the internal feature varables all_exp_kernel_recursive_sum_ and intensity_itegral_features_.
 * @param[in] data A collection of input sequences.
 */
	void Initialize(const std::vector<Sequence>& data);

/**
 * 	\brief Restores the optimization methods and regualrization into their default values.
 */
	void RestoreOptionToDefault();

/**
 * \brief Auxiliary function of sampling dimension for fast simulation of Hawkes process.
 * @param  lambda A column vector where each component is the intensity of the respective dimension.
 * @return        the dimension which can most likely explain the generation of the current event point.
 */
	unsigned AssignDim(const Eigen::VectorXd& lambda);

/**
 * \brief Auxiliary function of updating the intensity function of each dimension for fast simulation of Hawkes process.
 * @param[in] t              current given time.
 * @param[in] last_event_per_dim a column vector where each component is the last event of the respective dimension.
 * @param[out] expsum             cumulative influence between dimensions at time t. \f$\text{expsum(m,n)(t)} = \sum_{t^m_j < t}\exp(-\beta_{mn}(t - t^m_j)) = \exp(-\beta_{mn}(t - t^m_{n_m}))(1 + \text{expsum(m,n)}(t^m_{n_m}))\f$ where \f$n_m\f$ is the index of the last event on dimension m.
 */
	void UpdateExpSum(double t, const Eigen::VectorXd& last_event_per_dim, Eigen::MatrixXd& expsum);

public:

	/**
	 * \brief Optimization algorithms used to fit the standard Hawkes Process.
	 */
	enum OptMethod {	
						/**
						 * stochastic gradient descend.
						 */
						SGD, 
						/**
						 * [projected LBFGS](http://jmlr.csail.mit.edu/proceedings/papers/v5/schmidt09a/schmidt09a.pdf).
						 */
						PLBFGS 
				   };

	/**
	 * \brief Supported regularizations used to fit the standard Hawkes Process.
	 */
	enum Regularizer {
						/**
						 * Sparse L1 norm \f$\|\cdot\|_1\f$
						 */
						L1, 
						/**
						 * L22 norm \f$\|\cdot\|_2^2\f$
						 */
						L22, 
						/**
						 * Nuclear norm \f$\|\mathbf{A}\|_* = \sum_{i=1}^{\min(m,n)}\sigma_i\f$ where \f$sigma_i\f$ is the singular value of matrix \f$\mathbf{A}\f$
						 */
						NUCLEAR, 
						/**
						 * No regularization
						 */
						NONE
					};

	/**
	 * 	\brief Regularization coefficients.
	 */
	enum RegCoef {
					/**
					 * Regularization coefficient for \f$\|\boldsymbol{\lambda}_0\|\f$
					 */
					LAMBDA, 
					/**
					 * Regularization coefficient for \f$\|\mathbf{A}\|\f$
					 */
					BETA};

	/**
	 * \brief Options used to configure the fitting of the standard Hawkes Process.
	 */
	struct OPTION
	{
		/**
		 * Optimization method, which can be SGD or PLBFGS.
		 */
		OptMethod method;
		/**
		 * Type of regularization used for \f$\|\boldsymbol{\lambda}_0\|\f$
		 */
		Regularizer base_intensity_regularizer;
		/**
		 * Type of regularization used for \f$\|\mathbf{A}\|\f$
		 */
		Regularizer excitation_regularizer;
		/**
		 * Regularization coefficients value.
		 */
		std::map<RegCoef, double> coefficients;
		/**
		 * Initial learning rate of gradient descend.
		 */
		double ini_learning_rate;
		/**
		 * Coefficient to enforce the low-rank constraint.
		 */
		double rho;
		/**
		 * Upper bound estimation of the nuclear norm.
		 */
		double ub_nuclear;
		/**
		 * Maximum number of iterations.
		 */
		unsigned ini_max_iter;
	};

protected:
	/**
	 * \brief A configuration object which saves the optimization options.
	 */
	OPTION options_;

public:

/**
 * \brief The constructor 
 * @param[in] n the number of parameters in total.
 * @param[in] num_dims the number of dimensions in the process.
 * @param[in] Beta a D-by-D matrix, which stores the \f$\{\beta_{mn}\geq 0\}\f$ decaying rates of the exponential kernels. 
 */
	PlainHawkes(const unsigned& n, const unsigned& num_dims, const Eigen::MatrixXd& Beta) : IProcess(n, num_dims), Beta_(Beta), num_sequences_(0) 
	{
		options_.method = PLBFGS;
		options_.base_intensity_regularizer = NONE;
		options_.excitation_regularizer = NONE;
		options_.coefficients[LAMBDA] = 0;
		options_.coefficients[BETA] = 0;
		options_.ini_learning_rate = 1e-2;
		options_.rho = 1;
		options_.ub_nuclear = 25;
		options_.ini_max_iter = 1000;
		RNG_.SetState(314, 314);
	}

/**
 * \brief Maximum likelihood estimation for the model parameters.
 * @param[in] data    vectors of observed sequences.
 * @param[in] options data structure sotring different configuration for the optimization algorithm and the respective regularizations.
 */
	void fit(const std::vector<Sequence>& data, const OPTION& options);

/**
 * \brief Maximum likelihood estimation for the model parameters.	
 * @param[in] data           vectors of observed sequences.
 * @param[in] options        data structure sotring different configuration for the optimization algorithm and the respective regularizations.
 * @param[in] trueparameters a column vector storing the true parameters of the Hawkes process used to generate the set of observed sequences in data to compare.
 */
	void fit(const std::vector<Sequence>& data, const OPTION& options, const Eigen::VectorXd& trueparameters);

/**
 * \brief Negative loglikelihood of Hawkes process.
 * @param[out] objvalue negative loglikelihood. 
 * @param[out] gradient gradient of the parameters.
 */
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

/**
 * \brief Gradient on the \f$k\f$-th data sample.
 * @param[in] data sample index.
 * @param[out] gradient of the objective function w.r.t. the current data sample
 */
	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

/**
 * \brief Intensity function of the standard multi-dimensional Hawkes process. 
 * @param[in]  t             the given time t.
 * @param[in]  data          the sequence of past events.
 * @param[out]  intensity_dim a column vector where the n-th component is the intensity function of dimension n at the current given time t, that is,
 * \f{align}{
 * \text{intensity_dim[n]} = \lambda^n(t) = \lambda^n_0 + \sum_{m=1}^D\alpha_{mn}\sum_{t^m_j < t}\exp(-\beta_{mn}(t - t^m_j)).
 * \f}
 * @return               the total intensity value from all dimensions at the time t, that is, \f$\sum_{n=1}^D\lambda^n(t)\f$.
 */
	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

/**
 * \brief The upper bound of the intensity function between \f$[t, t + L]\f$.
 * @param[in]  t                   the given time t.
 * @param[in]  L                   the length we look into the future starting from the time t.
 * @param[in]  data                the sequence of past events.
 * @param[out]  intensity_upper_dim a column vector where the n-th component is the upper bound for the n-th dimension between \f$[t, t + L]\f$.
 * @return                     the total intensity upper bound from all dimensions, that is, \f$\sum_{n=1}^D\text{intensity_dim[n]}\f$.
 */
	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

/**
 * \brief The integral of the intensity function within the range \f$[\text{lower}, \text{upper}]\f$.
 * @param[in]  lower the inegeral lower bound.
 * @param[in]  upper the integral upper bound.
 * @param[in] data  the sequence of past events.
 * @return       the intensity integral \f$\int_{\text{lower}}^{\text{upper}}\lambda(t)dt\f$.
 */
	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

/**
 * \brief Predict the next event timing by the expectation \f$\int_{t_n}^\infty tf^*(t)dt\f$. Currently, we use the sample average by simulations to approximate the expectation since the conditional density \f$f^*(t)\f$ normally does not have an analytic form.
 * @param[in]  data            the sequence of past events.
 * @param[in]  num_simulations number of simulations we use to calculate the sample average.
 * @return                 the prediction of the next event timing.
 */
	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

/**
 * \brief Fast simulation method based on the property of the exponential triggering kernel using dynamic programming.
 *
 * The intensity function on the n-th dimension is defined as \f$\lambda^n(t) = \lambda^n_0 + \sum_{m=1}^D\alpha_{mn}\sum_{t^m_j < t}\exp(-\beta_{mn}(t - t^m_j))\f$. Let \f$A_{mn}(t) = \sum_{t^m_j < t}exp(-\beta_{mn}(t - t^m_j))\f$. Then, \f$A_{mn}(t) = \exp(-\beta_{mn}(t - t^n_i))A_{mn}(t^n_i) + \sum_{t^n_i\leq t^m_j < t}\exp(-\beta_{mn}(t - t^m_j))\f$.
 * @param[in] vec_T     a columne vector where each component is the observation window for each simulated sequence.
 * We can simulate multiple sequences. Each component of the column vector vec_T is the given observation window of the respective sequence, so the dimension of vec_T is equal to the number of simulated sequences in total.	 
 * @param[out] sequences a vector of simulated sequences.
 */
	void Simulate(const std::vector<double>& vec_T, std::vector<Sequence>& sequences);

/**
 * \brief Fast simulation method based on the property of the exponential triggering kernel using dynamic programming.
 * @param[in] n             number of events per simulated sequence.
 * @param[in] num_sequences number of simulated sequences.
 * @param[out] sequences a vector of simulated sequences.
 */
	void Simulate(const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences);

};

#endif