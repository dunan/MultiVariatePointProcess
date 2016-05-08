/**
 * \file HawkesLearningTriggeringKernel.h
 * \brief The class definition of HawkesLearningTriggeringKernel.
 */
#ifndef HAWKES_LEARNING_TRIGGERING_KERNEL_H
#define HAWKES_LEARNING_TRIGGERING_KERNEL_H

#include <vector>
#include <cmath>
#include <string>
#include <map>
#include "Process.h"
#include "Optimizer.h"
#include "Graph.h"

/**
 * \class HawkesLearningTriggeringKernel HawkesLearningTriggeringKernel.h "include/HawkesLearningTriggeringKernel.h"
 * \brief HawkesLearningTriggeringKernel implements the multivariate Hawkes process where the triggering kernel can be learned from the data.
 *
 * A multivariate Hawkes Process is a process where the occurrence of an event to a dimension will trigger more events on this dimension and other related dimensions in the near future. The intensity function of each dimension of the Hawkes process is defined as the following:
 * \f{align}{
 * 	\lambda^n(t) = \lambda_0^n + \sum_{m=1}^D\sum_{t^m_j < t}\gamma_{mn}(t, t^m_j),
 * \f}
 * where \f$\lambda_0^n\geq 0\f$ is the base intensity, \f$D\f$ is the number of dimensions, and \f$\gamma_{mn}(t, t^m_j)\geq 0\f$ captures the extent to which an event on dimension \f$m\f$ at the time \f$t^m_j\f$ can trigger an event on dimension \f$n\f$ at the time \f$t\f$. We parametrize \f$\gamma_{mn}(t, t^m_j) = \boldsymbol{k}_{mn}(t - t^m_j)^\top\boldsymbol{\alpha}_{mn}\f$. \f$\boldsymbol{k}_{mn}(t)\f$ is a column vector where the \f$l\f$-th component is defined as \f$\boldsymbol{k}_{mnl}(t)=\exp(-(\frac{\tau_l - t}{\sqrt{2}\sigma_l})^2)\f$, and \f$\tau_l\f$ and \f$\sigma_l\f$ are the parameters of the basis function.
 *
 * 
 */
class HawkesLearningTriggeringKernel : public IProcess
{

public:

	/**
	 * \brief Supported regularizations used to fit the standard Hawkes Process.
	 */
	enum Regularizer {
		/**
		 * L22 norm \f$\|\cdot\|_2^2\f$
		 */
		L22,
		/**
		 * GROUP LASSO \f$\sum_{m}\|\boldsymbol{\alpha}_{mn}\|_2\f$
		 * 
		 */ 
		GROUP, 
		/**
		 * No regularization
		 */
		NONE
	};

	/**
	 * 	\brief Regularization coefficients.
	 */
	enum RegCoef 
	{
		/**
		 * Regularization coefficient for \f$\|\boldsymbol{\lambda}_0\|\f$
		 */
		LAMBDA0, 
		/**
		 * Regularization coefficient for GROUP LASSO
		 * 
		 */
		LAMBDA
	};

	/**
	 * \brief Options used to configure the fitting of the general Hawkes Process with learned triggering kernels.
	 */
	struct OPTION
	{
		/**
		 * Type of regularization used for \f$\|\boldsymbol{\lambda}_0\|\f$
		 */
		Regularizer base_intensity_regularizer;
		/**
		 * Type of regularization used for learning the triggering kernel
		 */
		Regularizer excitation_regularizer;
		/**
		 * Regularization coefficients value.
		 */
		std::map<RegCoef, double> coefficients;	
	};

protected:

	const double PI = 3.14159265358979323846;

/**
 * \brief A column vector of size num_rbfs_. Each component of tau_ is the location of the respective RBF basis function.
 */
	Eigen::VectorXd tau_;
	
/**
 * \brief Total number of RBF basis functions.
 */
	unsigned num_rbfs_;
/**
 * \brief A column vector of size num_rbfs_. Each component of sigma_ is the bandwidth of the respective RBF basis function.
 */
	Eigen::VectorXd sigma_;
/**
 * \brief Helper variable defined as \f$\sqrt{2}\f$sigma_.
 */
	Eigen::VectorXd sqrt2sigma_;
/**
 * \brief Helper variable defined as \f$0.5\sqrt{2\pi}\f$sigma_.
 */
	Eigen::VectorXd sqrt2PIsigma_;
/**
 * \brief Helper variable defined as \f$\text{erfc}\bigg(\frac{tau\_}{\sqrt{2}\text{sigma_}}\bigg)\f$.
 */
	Eigen::VectorXd erfctau_sigma_;	
/**
 * \brief Temporal features associated with the intensity function.
 *
 * The log-likelihood of observing a collection of C sequences can be derived as the following:
 * \f{align}{
 * \sum_{n=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\sum_{i = 1}^{n_c}\bigg(\log(\lambda^n_0 + \sum_{m=1}^D\sum_{t^m_{j,c}<t^n_{i,c}}\underbrace{\boldsymbol{k}_{mn}(t^n_{i,c} - t^m_{j,c})^\top}_{\text{arrayK[n][c][i](m,:)}}\boldsymbol{\alpha}_{mn})\bigg) - T_c\lambda_0^n - \sum_{m=1}^D\sum_{t^m_{j,c} < T_c}\underbrace{\int_{t^m_j}^{T_c}\boldsymbol{k}_{mn}(t - t^m_{j,c})^\top dt}_{\text{arrayG[n][c](m,:)}}\cdot\boldsymbol{\alpha}_{mn}\bigg)\bigg\}.
 * \f}
 *
 * arrayK[n][c][i] is a \f$D\f$ by num_rbfs_ matrix which stores the influence from the event at the time \f$t^m_{j,c}\f$ on dimension \f$m\f$ to the event at the time \f$t^n_{i,c}\f$ on dimension \f$n\f$ in the sequence c.
 */
	std::vector<std::vector<std::vector<Eigen::MatrixXd> > > arrayK;
/**
 * \brief Temporal features derived from the integral of the intensity.
 *
 *  The log-likelihood of observing a collection of C sequences can be derived as the following:
 * \f{align}{
 * \sum_{n=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\sum_{i = 1}^{n_c}\bigg(\log(\lambda^n_0 + \sum_{m=1}^D\sum_{t^m_{j,c}<t^n_{i,c}}\underbrace{\boldsymbol{k}_{mn}(t^n_{i,c} - t^m_{j,c})^\top}_{\text{arrayK[n][c][i](m,:)}}\boldsymbol{\alpha}_{mn})\bigg) - T_c\lambda_0^n - \sum_{m=1}^D\sum_{t^m_{j,c} < T_c}\underbrace{\int_{t^m_j}^{T_c}\boldsymbol{k}_{mn}(t - t^m_{j,c})^\top dt}_{\text{arrayG[n][c](m,:)}}\cdot\boldsymbol{\alpha}_{mn}\bigg)\bigg\}.
 * \f}
 *
 * arrayG[n][c] is a \f$D\f$ by num_rbfs_ matrix which stores the integral of the influence from dimension \f$m\f$ to dimension \f$n\f$ in the sequence c.
 */
	std::vector<std::vector<Eigen::MatrixXd> > arrayG;
/**
 * \brief A column vector of length \f$C\f$ which is the total number of sequences. Each component records the observation window in the respective sequence.
 */
	Eigen::VectorXd observation_window_T_;
/**
 * \brief Total number of observed sequences
 */
	unsigned num_sequences_;
/**
 * \brief Initialize the temporal features [arrayK](@ref arrayK) and [arrayG](@ref arrayG) from the input sequences
 * @param[in] data input collection of sequences
 */
	void Initialize(const std::vector<Sequence>& data);
/**
 * \brief Initialize the temporal features [arrayK](@ref arrayK) and [arrayG](@ref arrayG) from the input sequences where the dependency structure among the dimensions is given.
 * @param[in] data input collection of sequences
 */
	void InitializeWithGraph(const std::vector<Sequence>& data);
/**
 * \brief Initialize all helper variables
 */
	void InitializeConstants();
/**
 * \brief Post process the learned dependency structure
 */
	void PostProcessing();
/**
 * \brief Compute the negative loglikelihood.
 * @param[out] objvalue negative loglikelihood. 
 * @param[out] gradient gradient of the parameters.
 */
	void GetNegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);
/**
 * \brief A configuration object which saves the optimization options.
 */
	OPTION options_;
/**
 * \brief A graph object represents the dependency structure among the dimensions. 
 */
	const Graph* graph_;

public:

	/**
	 * \brief The constructor 
	 * @param[in] n the number of parameters in total.
	 * @param[in] num_dims the number of dimensions in the process.
	 * @param[in] tau the location of each basis function.
	 * @param[in] sigma the bandwidth of each basis function.
	 */
	HawkesLearningTriggeringKernel(const unsigned& n, const unsigned& num_dims, const Eigen::VectorXd& tau, const Eigen::VectorXd& sigma) : IProcess(n, num_dims), tau_(tau), sigma_(sigma), num_sequences_(0), graph_(NULL)
	{
		HawkesLearningTriggeringKernel::InitializeConstants();
	}

	/**
	 * \brief The constructor 
	 * @param[in] n the number of parameters in total.
	 * @param[in] num_dims the number of dimensions in the process.
	 * @param[in] graph the graph object representing the dependency structure among the dimensions.
	 * @param[in] tau the location of each basis function.
	 * @param[in] sigma the bandwidth of each basis function.
	 */
	HawkesLearningTriggeringKernel(const unsigned& n, const unsigned& num_dims, const Graph* graph, const Eigen::VectorXd& tau, const Eigen::VectorXd& sigma) : IProcess(n, num_dims), tau_(tau), sigma_(sigma), num_sequences_(0), graph_(graph)
	{
		HawkesLearningTriggeringKernel::InitializeConstants();
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
 	 * - \sum_{n=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\sum_{i = 1}^{n_c}\bigg(\log(\lambda^n_0 + \sum_{m=1}^D\sum_{t^m_{j,c}<t^n_{i,c}}\underbrace{\boldsymbol{k}_{mn}(t^n_{i,c} - t^m_{j,c})^\top}_{\text{arrayK[n][c][i](m,:)}}\boldsymbol{\alpha}_{mn})\bigg) - T_c\lambda_0^n - \sum_{m=1}^D\sum_{t^m_{j,c} < T_c}\underbrace{\int_{t^m_j}^{T_c}\boldsymbol{k}_{mn}(t - t^m_{j,c})^\top dt}_{\text{arrayG[n][c](m,:)}}\cdot\boldsymbol{\alpha}_{mn}\bigg)\bigg\}.
 	 * \f}
	 * 
	 * @param[out] objvalue negative loglikelihood. 
 	 * @param[out] gradient gradient of the parameters.
 	*/
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

	/**
	 * \brief Visualize the learned triggering kernel between dimension dim_m and dimension dim_n.
	 * @param dim_m the influential dimension
	 * @param dim_n the influenced dimension
	 * @param T     observation window.
	 * @param delta unit step size to partition the time line.
	 */
	void PlotTriggeringKernel(const unsigned& dim_m, const unsigned& dim_n, const double& T, const double& delta);

};


#endif