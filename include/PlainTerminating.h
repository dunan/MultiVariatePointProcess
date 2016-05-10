/**
 * \file PlainTerminating.h
 * \brief The class definition of PlainTerminating implementing the Terminating process.
 */
#ifndef PLAIN_TERMINATING_H
#define PLAIN_TERMINATING_H

#include <vector>
#include <string>
#include <map>
#include "Process.h"
#include "Optimizer.h"
#include "Graph.h"

/**
 * \class PlainTerminating PlainTerminating.h "include/PlainTerminating.h"
 * \brief PlainTerminating implements the multivariate terminating process.
 *
 * The Multivariate Terminating Point Process is an \f$D\f$-dimensional temporal point process with the conditional intensity function of each dimension \f$d\f$ is given by \f$\lambda_d^*(t) = \mathbb{I}\{N_d(t)\leq 1\}\cdot g(t)\f$ where \f$N_d(t)\f$ is the number of events on the dimension \f$d\f$, \f$g(t)\f$ is a non-negative function, and \f$\mathbb{I}\{{\cdot}\}\f$ is the indicator function. The Multivariate Terminating Process instantiates the continuous-time information diffusion model. In this class, we assume the pairwise diffusion time conforms to an exponential distribution, that is, \f$f_{ji}(t) = \alpha_{ji}\f$. Check out the following papers for more details. 
 * - [Uncovering the Temporal Dynamics of Diffusion Networks](https://www.mpi-sws.org/~manuelgr/pubs/netrate-icml11.pdf). M. Gomez-Rodriguez, D. Balduzzi, B. Schölkopf. The 28th International Conference on Machine Learning (ICML), 2011.
 * - [Learning Networks of Heterogeneous Influence](http://www.cc.gatech.edu/~ndu8/pdf/DuSonAleYua-NIPS-2012.pdf). Nan Du, Le Song, Alex Smola, and Ming Yuan. Neural Information Processing Systems (NIPS), 2012.
 */
class PlainTerminating : public IProcess
{

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
						 * No regularization
						 */
						NONE
					};

	/**
	 * 	\brief Regularization coefficients.
	 */
	enum RegCoef {

					/**
					 * Regularization coefficient for \f$\|\mathbf{A}\|\f$
					 */
					LAMBDA

				 };

	/**
	 * \brief Options used to configure the fitting of the terminating point process.
	 */
	struct OPTION
	{
		/**
		 * Optimization method, which can be SGD or PLBFGS.
		 */
		OptMethod method;
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
 * \brief the temporal features associated with the intensity.
 *
 * The log-likelihood of observing a collection of C sequences can be derived as the following:
 * \f{align}{
 * \sum_{i=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\log(\sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j<t^c_i)}_{\text{arrayK[i]}(c, j)}\alpha_{ji}) - \sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)(t^c_i - t^c_j)}_{\text{arrayG}[i](c,j)}\alpha_{ji}\bigg)\bigg\},
 * \f}
 * where \f$\alpha_{ji}\f$ is the pairwise infection risk from node \f$j\f$ to node \f$i\f$. \f$\text{arrayK[i]}(c, j)\f$ indicates whether node \f$j\f$ is the infecting parent of node \f$i\f$ in the sequence \f$c\f$.
 */
	std::vector<Eigen::MatrixXd> arrayK;

/**
 * \brief Intergral of the intensity.
 *
 * The log-likelihood of observing a collection of C sequences can be derived as the following:
 * \f{align}{
 * \sum_{i=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\log(\sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j<t^c_i)}_{\text{arrayK[i]}(c, j)}\alpha_{ji}) - \sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)(t^c_i - t^c_j)}_{\text{arrayG}[i](c,j)}\alpha_{ji}\bigg)\bigg\},
 * \f}
 * where \f$\alpha_{ji}\f$ is the pairwise infection risk from node \f$j\f$ to node \f$i\f$. \f$\text{arrayG[i]}(c, j)\f$ is the time duration between the infection time \f$t^c_j\f$ and \f$t^c_i\f$ in the sequence \f$c\f$. 
 */
	std::vector<Eigen::MatrixXd> arrayG;

/**
 * \brief a column vector of length \f$C\f$ which is the total number of sequences. Each component records the observation window in the respective sequence.
 */
	Eigen::VectorXd observation_window_T_;

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
 * \brief Initialize the temporal features [arrayK](@ref arrayK) and [arrayG](@ref arrayG) from the input sequences where the network structure among the nodes
 *  is given.
 * @param[in] data input collection of sequences
 */
	
	void InitializeWithGraph(const std::vector<Sequence>& data);
/**
 * \brief Post process the learned dependency structure
 */
	void PostProcessing();

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
	 */
	PlainTerminating(const unsigned& n, const unsigned& num_dims) : IProcess(n, num_dims), num_sequences_(0), graph_(NULL) 
	{
		options_.method = PLBFGS;
		options_.excitation_regularizer = NONE;
		options_.coefficients[LAMBDA] = 0;
	}

	/**
	 * \brief The constructor 
	 * @param[in] n the number of parameters in total.
	 * @param[in] num_dims the number of dimensions in the process.
	 * @param[in] graph the graph object representing the dependency structure among the dimensions.
	 */
	PlainTerminating(const unsigned& n, const unsigned& num_dims, const Graph* graph) : IProcess(n, num_dims), num_sequences_(0), graph_(graph)
	{
		options_.method = PLBFGS;
		options_.excitation_regularizer = NONE;
		options_.coefficients[LAMBDA] = 0;
	}


	/**
 	 * \brief Maximum likelihood estimation for the model parameters.
 	 * @param[in] data    vectors of observed sequences.
 	 * @param[in] options data structure sotring different configuration for the optimization algorithm and the respective regularizations.
 	 */
	void fit(const std::vector<Sequence>& data, const OPTION& options);

	/**
	 * \brief Negative loglikelihood of multivariate terminating point process.
	 *
	 * \f{align}{
 	 * \sum_{i=1}^D\bigg\{\frac{1}{C}\sum_{c=1}^C\bigg(\log(\sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j<t^c_i)}_{\text{arrayK[i]}(c, j)}\alpha_{ji}) - \sum_{j\neq i}\underbrace{\mathbb{I}(t^c_j < t^c_i)(t^c_i - t^c_j)}_{\text{arrayG}[i](c,j)}\alpha_{ji}\bigg)\bigg\},
 	 * \f}
 	 * where \f$\alpha_{ji}\f$ is the pairwise infection risk from node \f$j\f$ to node \f$i\f$.
	 * @param objvalue negative loglikelihood. 
	 * @param gradient gradient of the parameters.
	 */
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

/**
 * \brief Intensity function of each node.
 *
 * The intensity function of each node is defined as \f$\lambda^*_{c,i}(t) = \sum_{j\neq i}\mathbb{I}(t^c_j < t)\alpha_{ji}\f$.
 * @param[in]  t             the current given time.
 * @param[in]  data          sequence of past events.
 * @param[out]  intensity_dim a column vector of size num_dims_ where each component stores the intensity value of the respetive dimension at time t given the past sequence in data.
 * @return               the summation of the intensity value from each dimension. 
 */
	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

};

#endif