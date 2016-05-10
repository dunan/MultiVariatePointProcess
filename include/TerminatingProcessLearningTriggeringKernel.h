/**
 * \file TerminatingProcessLearningTriggeringKernel.h
 * \brief The class definition of TerminatingProcessLearningTriggeringKernel.
 */
#ifndef TERMINATING_PROCESS_LEARNING_TRIGGERING_KERNEL_H
#define TERMINATING_PROCESS_LEARNING_TRIGGERING_KERNEL_H

#include <vector>
#include <cmath>
#include <string>
#include <functional>
#include <map>
#include "Process.h"
#include "Optimizer.h"
#include "Graph.h"

/**
 * \class TerminatingProcessLearningTriggeringKernel TerminatingProcessLearningTriggeringKernel.h "include/TerminatingProcessLearningTriggeringKernel.h"
 * \brief TerminatingProcessLearningTriggeringKernel implements the multivariate terminating process where the pairwise infection risk function can be learned from the data.
 *
 * The Multivariate Terminating Point Process is an \f$D\f$-dimensional temporal point process with the conditional intensity function of each dimension \f$d\f$ is given by \f$\lambda_d^*(t) = \mathbb{I}\{N_d(t)\leq 1\}\cdot g(t)\f$ where \f$N_d(t)\f$ is the number of events on the dimension \f$d\f$, \f$g(t)\f$ is a non-negative function, and \f$\mathbb{I}\{{\cdot}\}\f$ is the indicator function. The Multivariate Terminating Process instantiates the continuous-time information diffusion model with general pairwise infection risk functions \f$\gamma_{ji}(t, t^c_j)\f$, so we have \f$g(t) = \sum_{j\neq i}\mathbb{I}(t^c_j < t)\gamma_{ji}(t, t^c_j)\f$ in a sequence \f$c\f$.
 *
 * Check out the following papers for more details.
 * - [Learning Networks of Heterogeneous Influence](http://www.cc.gatech.edu/~ndu8/pdf/DuSonAleYua-NIPS-2012.pdf). Nan Du, Le Song, Alex Smola, and Ming Yuan. Neural Information Processing Systems (NIPS), 2012.
 * - [Uncover Topic-Sensitive Information Diffusion Networks](http://www.cc.gatech.edu/~ndu8/pdf/DuSonWooZha-aistats-2013.pdf). Nan Du, Le Song, Hyenkyun Woo, and Hongyuan Zha. Sixteenth International Conference on Artificial Intelligence and Statistics (AISTATS) , Apr. 29 - May 1, 2013, Scottsdale, AZ, USA.
 */
class TerminatingProcessLearningTriggeringKernel : public IProcess
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
 * \sum_{i=1}^D\bigg\{\frac{1}{C}\sum_{c = 1}^C\bigg(\log(\sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)\boldsymbol{k}_{ji}(t^c_i - t^c_j)^\top}_{\text{arrayK}[i][c](j,:)}\cdot\boldsymbol{\alpha}_{ji}) - \sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)\int_{t^c_j}^{t^c_i}\boldsymbol{k}_{ji}(t - t^c_j)^\top}_{\text{arrayG[i]}[c](j,:)} dt\cdot\boldsymbol{\alpha}_{ji}\bigg)\bigg\}
 * \f}
 *
 * arrayK[i][c] is a \f$D\f$ by num_rbfs_ matrix which stores the influence from the event at the time \f$t^c_{j}\f$ on dimension \f$j\f$ to the event at the time \f$t^c_{i}\f$ on dimension \f$i\f$ in the sequence c. 
 */
	std::vector<std::vector<Eigen::MatrixXd> > arrayK;
/**
 * \brief Temporal features derived from the integral of the intensity.
 *
 * The log-likelihood of observing a collection of C sequences can be derived as the following:
 * \f{align}{
 * \sum_{i=1}^D\bigg\{\frac{1}{C}\sum_{c = 1}^C\bigg(\log(\sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)\boldsymbol{k}_{ji}(t^c_i - t^c_j)^\top}_{\text{arrayK}[i][c](j,:)}\cdot\boldsymbol{\alpha}_{ji}) - \sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)\int_{t^c_j}^{t^c_i}\boldsymbol{k}_{ji}(t - t^c_j)^\top}_{\text{arrayG[i]}[c](j,:)} dt\cdot\boldsymbol{\alpha}_{ji}\bigg)\bigg\}
 * \f}
 *
 * arrayG[i][c] is a \f$D\f$ by num_rbfs_ matrix which stores the integral of the influence from dimension dimension \f$j\f$ to dimension \f$i\f$ in the sequence c. 
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
	TerminatingProcessLearningTriggeringKernel(const unsigned& n, const unsigned& num_dims, const Eigen::VectorXd& tau, const Eigen::VectorXd& sigma) : IProcess(n, num_dims), tau_(tau), sigma_(sigma), num_sequences_(0), graph_(NULL)
	{
		options_.excitation_regularizer = NONE;
		options_.coefficients[LAMBDA] = 0;
		num_rbfs_ = tau_.size();

		sqrt2sigma_ = sqrt(2) * sigma_.array();
		sqrt2PIsigma_ = 0.5 * sqrt(2 * PI) * sigma_.array();
		erfctau_sigma_ = (tau_.array() / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc));

	}

	/**
	 * \brief The constructor 
	 * @param[in] n the number of parameters in total.
	 * @param[in] num_dims the number of dimensions in the process.
	 * @param[in] graph the graph object representing the dependency structure among the dimensions.
	 * @param[in] tau the location of each basis function.
	 * @param[in] sigma the bandwidth of each basis function.
	 */
	TerminatingProcessLearningTriggeringKernel(const unsigned& n, const unsigned& num_dims, const Graph* graph, const Eigen::VectorXd& tau, const Eigen::VectorXd& sigma) : IProcess(n, num_dims), tau_(tau), sigma_(sigma), num_sequences_(0), graph_(graph)
	{
		options_.excitation_regularizer = NONE;
		options_.coefficients[LAMBDA] = 0;
		num_rbfs_ = tau_.size();

		sqrt2sigma_ = sqrt(2) * sigma_.array();
		sqrt2PIsigma_ = 0.5 * sqrt(2 * PI) * sigma_.array();
		erfctau_sigma_ = (tau_.array() / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc));
	}

	/**
 	 * \brief Maximum likelihood estimation for the model parameters.
 	 * @param[in] data    vectors of observed sequences.
 	 * @param[in] options data structure sotring different configuration for the optimization algorithm and the respective regularizations.
 	 */
	void fit(const std::vector<Sequence>& data, const OPTION& options);

	/**
	 * \brief Negative loglikelihood of multivariate terminating point process learning triggering kernels
	 *
	 * \f{align}{
 	 * \sum_{i=1}^D\bigg\{\frac{1}{C}\sum_{c = 1}^C\bigg(\log(\sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)\boldsymbol{k}_{ji}(t^c_i - t^c_j)^\top}_{\text{arrayK}[i][c](j,:)}\cdot\boldsymbol{\alpha}_{ji}) - \sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)\int_{t^c_j}^{t^c_i}\boldsymbol{k}_{ji}(t - t^c_j)^\top}_{\text{arrayG[i]}[c](j,:)} dt\cdot\boldsymbol{\alpha}_{ji}\bigg)\bigg\}
 	 * \f}
 	 *  
	 * @param[out] objvalue negative loglikelihood. 
	 * @param[out] gradient gradient of the parameters.
	 */
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	//  Return the stochastic gradient on the random sample k.
	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

	/**
	 * \brief Intensity function of each dimension(or node)
	 *
	 * \f{align}{
	 * 	\lambda_i^*(t) = \mathbb{I}\{N_i(t)\leq 1\}\sum_{j\neq i}\mathbb{I}(t^c_j < t)\gamma_{ji}(t, t^c_j)
	 * \f}
	 * in a given sequence \f$c\f$ where \f$\gamma_{ji}(t, t^c_j) = \boldsymbol{k}_{ji}(t - t^c_j)^\top\boldsymbol{\alpha}_{ji}\f$. \f$\boldsymbol{k}_{ji}(t)\f$ is a column vector where the \f$l\f$-th component is defined as \f$\boldsymbol{k}_{jil}(t)=\exp(-(\frac{\tau_l - t}{\sqrt{2}\sigma_l})^2)\f$, and \f$\tau_l\f$ and \f$\sigma_l\f$ are the parameters of the basis function.
	 * @param  t             current given time.
	 * @param  data          the given sequence of the past events until time t.
	 * @param  intensity_dim a column vector of size num_dims_ where each component stores the intensity value of the respetive dimension at time t given the past sequence in data.
	 * @return               the summation of the intensity value from each dimension. 
	 */
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