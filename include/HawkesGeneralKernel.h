/**
 * \file HawkesGeneralKernel.h
 * \brief The class definition of HawkesGeneralKernel for [Hawkes](@ref PlainHawkes) process with customized triggering kernels.
 */
#ifndef HAWKES_GENERAL_KERNEL
#define HAWKES_GENERAL_KERNEL

#include <vector>
#include <map>
#include "Process.h"
#include "TriggeringKernel.h"
#include "SimpleRNG.h"

/**
 * \class HawkesGeneralKernel HawkesGeneralKernel.h "include/HawkesGeneralKernel.h"
 * \brief HawkesGeneralKernel implements the multivariate Hawkes process with customized triggering kernels.
 *
 * A multivariate Hawkes Process is a process where the occurrence of an event to a dimension will trigger more events on this dimension and other related dimensions in the near future. The intensity function of each dimension of the Hawkes process is generally defined as the following:
 * \f{align}{
 * 	\lambda^n(t) = \lambda_0^n + \sum_{m=1}^D\alpha_{mn}\sum_{t^m_j < t}\gamma(t - t^m_j),
 * \f}
 * where \f$\lambda_0^n\geq 0\f$ is the base intensity, \f$D\f$ is the number of dimensions, \f$\alpha_{mn}\geq 0\f$, and the triggering kernel \f$\gamma(t - t^m_j)\f$  captures the extent to which an event on dimension m at the time \f$t^m_j\f$ can trigger an event on dimension n in the near future. Normally, in the [standard Hawkes process](@ref PlainHawkes), we have \f$\gamma(t - t^m_j) = \exp(-\beta_{mn}(t - t^m_j))\f$. However, in more general cases, the form of the triggering kernel can be formulated to catpure the phenomena of interest. The collection of \f$\{\alpha_{mn}\}\f$ can be represented as a matrix \f$\mathbf{A}(m,n) = \alpha_{mn}\f$, and the collection of \f$\{\lambda_0^n\}\f$ can be represented as a column vector \f$\boldsymbol{\lambda}_0\f$. 
 */

class HawkesGeneralKernel : public IProcess
{

protected:

/**
 * \brief the temporal features associated with the intensity
 *
 * The log-likelihood of observing a collection of C sequences can be derived as the following:
 * \f{align}{
 * \sum_{n=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\sum_{i = 1}^{n_c}\bigg(\log(\lambda^n_0 + \sum_{m=1}^D\alpha_{mn}\underbrace{\sum_{t^m_{j,c}<t^n_{i,c}}\gamma_{mn}(t^n_{i,c} - t^m_{j,c})}_{\text{arrayK[n][c](i,m)}})\bigg) - T_c\lambda_0^n - \sum_{m=1}^D\alpha_{mn}\underbrace{\sum_{t^m_{j,c} < T_c}\int_{t^m_j}^{T_c}\gamma_{mn}(t - t^m_{j,c})dt)}_{\text{arrayG[n](c,m)}}\bigg)\bigg\}.
 * \f}
 *
 * arrayK[n][c] is an \f$n_c\f$ by \f$D\f$ matrix where \f$n_c\f$ is the number of events on the nth dimension in the sequence c. arrayK[n][c](i,m) stores the cumulative influence of the past events on dimension \f$m\f$ in the sequence \f$c\f$ to the occurence of the \f$i\f$th event on dimension \f$n\f$.
 */
	std::vector<std::vector<Eigen::MatrixXd> > arrayK;

/**
 * \brief summation of the integral of the triggering kernels
 *
 * The log-likelihood of observing a collection of C sequences can be derived as the following:
 * \f{align}{
 * \sum_{n=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\sum_{i = 1}^{n_c}\bigg(\log(\lambda^n_0 + \sum_{m=1}^D\alpha_{mn}\underbrace{\sum_{t^m_{i,c}<t^n_{i,c}}\gamma_{mn}(t^n_{i,c} - t^m_{i,c})}_{\text{arrayK[n][c](i,m)}})\bigg) - T_c\lambda_0^n - \sum_{m=1}^D\alpha_{mn}\underbrace{\sum_{t^m_{j,c} < T_c}\int_{t^m_j}^{T_c}\gamma_{mn}(t - t^m_{j,c})dt)}_{\text{arrayG[n](c,m)}}\bigg)\bigg\}.
 * \f}
 *
 * arrayG[n] is an \f$C\f$ by \f$D\f$ matrix where \f$n_c\f$ is the number of events on the nth dimension in the sequence c. arrayK[n][c](i,m) stores the cumulative influence of the past events on dimension \f$m\f$ in the sequence \f$c\f$ to the occurence of the \f$i\f$th event on dimension \f$n\f$.
 */
	std::vector<Eigen::MatrixXd> arrayG;

/**
 *	\brief a \f$D\f$ by \f$D\f$ grid of [triggering-kernels](@ref TriggeringKernel)
 *	
 */
	std::vector<std::vector<TriggeringKernel*> > triggeringkernels_;

/**
 * \brief a column vector of length \f$C\f$ which is the total number of sequences. Each component records the observation window in the respective sequence.
 */
	Eigen::VectorXd observation_window_T_;

/**
 * \brief internal implementation for random number generator
 */
	SimpleRNG RNG_;

/**
 * \brief total number of observed sequences
 */
	unsigned num_sequences_;

/**
 * \brief initialize the temporal features [arrayK](@ref arrayK) and [arrayG](@ref arrayG) from the input sequences
 * @param[in] data input collection of sequences
 */
	void Initialize(const std::vector<Sequence>& data);

/**
 * \brief restore to the default optimization configuration
 */
	void RestoreOptionToDefault();

public:
/**
 * \brief Supported regularizations used to fit the standard Hawkes Process
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
					LAMBDA0, 
					/**
					 * Regularization coefficient for \f$\|\mathbf{A}\|\f$
					 */
					LAMBDA
				};

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
	 * \brief Options used to configure the fitting of the general Hawkes Process with customized triggering kernels.
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
 * @param[in] triggeringkernels a D-by-D grid of [triggering-kernels](@ref TriggeringKernel).
 */
	HawkesGeneralKernel(const unsigned& n, const unsigned& num_dims, const std::vector<std::vector<TriggeringKernel*> >& triggeringkernels) : IProcess(n, num_dims), num_sequences_(0) 
	{
		options_.method = PLBFGS;
		options_.base_intensity_regularizer = NONE;
		options_.excitation_regularizer = NONE;
		options_.coefficients[LAMBDA0] = 0;
		options_.coefficients[LAMBDA] = 0;

		RNG_.SetState(314, 314);

		triggeringkernels_ = triggeringkernels;
	}

/**
 * \brief Maximum likelihood estimation for the model parameters.
 * @param[in] data    vectors of observed sequences.
 * @param[in] options data structure sotring different configuration for the optimization algorithm and the respective regularizations.
 */
	void fit(const std::vector<Sequence>& data, const OPTION& options);

/**
 * \brief Negative loglikelihood of general Hawkes process
 * 
 * \f{align}{
 * -\sum_{n=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\sum_{i = 1}^{n_c}\bigg(\log(\lambda^n_0 + \sum_{m=1}^D\alpha_{mn}\underbrace{\sum_{t^m_{i,c}<t^n_{i,c}}\gamma_{mn}(t^n_{i,c} - t^m_{i,c})}_{\text{arrayK[n][c](i,m)}})\bigg) - T_c\lambda_0^n - \sum_{m=1}^D\alpha_{mn}\underbrace{\sum_{t^m_{j,c} < T_c}\int_{t^m_j}^{T_c}\gamma_{mn}(t - t^m_{j,c})dt)}_{\text{arrayG[n](c,m)}}\bigg)\bigg\}.
 * \f}
 * @param[out] objvalue negative loglikelihood. 
 * @param[out] gradient gradient of the parameters.
 */
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

};

#endif