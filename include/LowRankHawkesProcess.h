/**
 * \file LowRankHawkesProcess.h
 * \brief The class definition of LowRankHawkesProcess implementing the low-rank Hawkes process.
 */
#ifndef LOW_RANK_HAWKES_PROCESS_H
#define LOW_RANK_HAWKES_PROCESS_H
#include <vector>
#include <string>
#include <map>
#include <Eigen/Sparse>
#include "Process.h"
#include "Optimizer.h"

/**
 * \class LowRankHawkesProcess LowRankHawkesProcess.h "include/LowRankHawkesProcess.h"
 * \brief LowRankHawkesProcess implements the standard multivariate Hawkes process.
 * 
 * The Low-rank Hawkes Process is an $mn$-dimensional Hawkes process arranged in an \f$m\f$-by-\f$n\f$ grid. For instance, we have \f$m\f$ users and \f$n\f$ items. The conditional intensity function of the entry \f$(u,i)\f$ between user \f$u\f$ and item \f$i\f$ is \f$\lambda^{u,i}(t) = \boldsymbol{\Lambda}_0(u,i) + \boldsymbol{A}(u,i)\sum_{t^{u,i}_k < t}\exp(-\beta_{u,i}(t - t^{u,i}_k))\f$ where \f$\boldsymbol{\Lambda}_0\f$ and \f$\boldsymbol{A}\f$ are the low-rank base intensity matrix and the self-exciting matrix, respectively.
 * Check out the following paper for more details.
 * - [Time-Sensitive Recommendation From Recurrent User Activities](http://www.cc.gatech.edu/~ndu8/pdf/DuWangHeSong-NIPS-2015.pdf). Nan Du, Yichen Wang, Niao He, and Le Song. Neural Information Processing Systems (NIPS), 2015, Montreal, Quebec, Canada.
 */
class LowRankHawkesProcess : public IProcess
{

protected:

/**
 * 	\brief A column vector of size \f$|\Omega|\f$ where each component is the decaying rate of the respective exponential triggering kernel for each observed user-item pair \f$(u,i)\in\Omega\f$. 
 */
	Eigen::VectorXd beta_;

/**
 * \brief Temporal features associated with each event. 
 * 
 * Suppose we have observed a collection \f$\Omega = \{(u,i)\}\f$ of user-item pairs. Each user-item pair \f$(u,i)\f$ induces a sequence of \f$n_{u,i}\f$ events. The total number of events is thus \f$N = \sum_{(u,i)\in\Omega}n_{u,i}\f$. event_intensity_features_ is a column vector of size \f$N\f$, which can be regarded as the sequential concatenation of each sequence from the respective user-item pair. Each component of event_intensity_features_ is the summation of the influence from the past events to the current event in the respective sequence, that is, \f$\sum_{t^{u,i}_k < t^{u,i}_j}\exp(-\beta_{u,i}(t^{u,i}_j - t^{u,i}_k))\f$. 
 */
	Eigen::VectorXd event_intensity_features_;

/**
 * \brief Temporal features associated with each user-item pair.
 *
 * Suppose we have observed a collection \f$\Omega = \{(u,i)\}\f$ of user-item pairs. integral_intensity_features_ is a column vector of size \f$|\Omega|\f$ where each component stores \f$\sum_{t^{u,i}_k < T^{u,i}}\int_{t^{u,i}_k}^{T^{u,i}}\lambda^{u,i}(t)dt\f$ for the respective user-item pair \f$(u,i)\f$ where \f$T^{u,i}\f$ is the corresponding observation window.
 */
	Eigen::VectorXd integral_intensity_features_;

/**
 * \brief A sparse bitmap mapping matrix.
 *
 * pair_event_map_ is a sparse binary matrix of size \f$|\Omega|\f$ by \f$\sum_{(u,i)\in\Omega}n_{u,i}\f$. For each pair \f$(u,i)\f$, the \f$n_{u,i}\f$ entries (out of the total \f$\sum_{(u,i)\in\Omega}n_{u,i}\f$ columns) in the respective row are set to be one. Because each row is of size \f$\sum_{(u,i)\in\Omega}n_{u,i}\f$ which is the total number of events induced from all observed user-item pairs, the non-zero entries in each row marks the location of the events belonging to the respective \f$(u,i)\f$ pair.
 */
	Eigen::SparseMatrix<double> pair_event_map_;

/**
 * 	\brief observation_window_T_ stores the observation window for each sequence.
 *
 * 	For each sequence induced from the pair \f$(u,i)\f$, \f$\text{observation_window_T_}(c) = T^{u,i}\f$.
 */
	Eigen::VectorXd observation_window_T_;

/**
 * \brief Stores the index of each observed user-item pair in the total \f$m\f$ by \f$n\f$ grid.
 */
	Eigen::VectorXi observed_idx_;

/**
 * \brief Total number of rows(or users).
 */
	unsigned num_rows_;

/**
 * \brief Total number of columns(or items).
 */
	unsigned num_cols_;


/**
 * \brief Initialize the temporal features [event_intensity_features_](@ref event_intensity_features_) and [integral_intensity_features_](@ref integral_intensity_features_).
 * @param[in] data collection of input sequences induced from each observed user-item pair.
 */
	void Initialize(const std::vector<Sequence>& data);

/**
 * \brief Helper function returns the respective index in a column vector of the given (row, column) pair.
 * @param[in]  i row index.
 * @param[in]  j column index.
 * @return   the index of the pair in a column vector.
 */
	unsigned Vec2Ind(const unsigned& i, const unsigned& j);

/**
 * \brief Helper function returns the respective (row, column) pair given the index in the corresponding column vector.
 * @param[in] ind index in the long column vector.
 * @param[out] i   returned row index.
 * @param[out] j   returned column index.
 */
	void Ind2Vec(const unsigned& ind, unsigned& i, unsigned& j);

public:

	/**
	 * 	\brief Regularization coefficients.
	 */
	enum RegCoef {
					/**
					 * Regularization coefficient for \f$\|\boldsymbol{\Lambda}_0\|\f$
					 */
					LAMBDA0, 
					/**
					 * Regularization coefficient for \f$\|\mathbf{A}\|\f$
					 */
					LAMBDA
				};

	/**
	 * \brief Options used to configure the fitting of the Low-rank Hawkes process.
	 */
	struct OPTION
	{
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
		 * Upper bound estimation of the nuclear norm of the base intensity matrix
		 */
		double ub_nuclear_lambda0;
		/**
		 * Upper bound estimation of the nuclear norm of the excitation matrix
		 */
		double ub_nuclear_alpha;
		/**
		 * Maximum number of iterations.
		 */
		unsigned ini_max_iter;		
	};

protected:

	/**
	 * \brief A configuration object which saves the optimization options.
	 */
	LowRankHawkesProcess::OPTION options_;

private:

	class ExpectationHandler : public FunctionHandler
	{
		private:
			
			unsigned uid_;
			
			unsigned itemid_;

			Sequence sequence_;

			LowRankHawkesProcess& parent_;

		public:

			ExpectationHandler(unsigned u, unsigned i, const Sequence& sequence, LowRankHawkesProcess& parent) : uid_(u), itemid_(i), sequence_(sequence), parent_(parent){}
			
			virtual void operator()(const Eigen::VectorXd& t, Eigen::VectorXd& y);
	};

public:

/**
 * \brief The constructor.
 *
 * @param[in] num_rows the total number of rows(or users).
 * @param[in] num_dims the total number of columns(or items).
 * @param[in] beta a column vector of size \f$|\Omega|\f$ where each component is the decaying rate of the respective exponential triggering kernel for each observed user-item pair \f$(u,i)\in\Omega\f$. 
 */
	LowRankHawkesProcess(const unsigned& num_rows, const unsigned& num_cols, const Eigen::VectorXd& beta) : IProcess(2 * num_rows * num_cols, num_rows * num_cols), beta_(beta), num_rows_(num_rows), num_cols_(num_cols)
	{
		options_.coefficients[LowRankHawkesProcess::LAMBDA0] = 0;
		options_.coefficients[LowRankHawkesProcess::LAMBDA] = 0;
		options_.ini_learning_rate = 1e-2;
		options_.rho = 1;
		options_.ub_nuclear_lambda0 = 25;
		options_.ub_nuclear_alpha = 25;
		options_.ini_max_iter = 1000;	
	}

/**
 * \brief Maximum likelihood estimation for the model parameters.
 * @param[in] data    vectors of observed sequences.
 * @param[in] options data structure containing different configurations for the optimization algorithm and the respective regularizations.
 */
	void fit(const std::vector<Sequence>& data, const OPTION& options);

/**
 * \brief Maximum likelihood estimation for the model parameters.
 * @param[in] data        vectors of observed sequences.
 * @param[in] options     data structure containing different configurations for the optimization algorithm and the respective regularizations.
 * @param[in] TrueLambda0 ground truth \f$\boldsymbol{\Lambda}_0\f$.
 * @param[in] TrueAlpha   ground truth \f$\boldsymbol{A}\f$.
 */
	void fit(const std::vector<Sequence>& data, const LowRankHawkesProcess::OPTION& options, const Eigen::MatrixXd& TrueLambda0, const Eigen::MatrixXd& TrueAlpha);

/**
 * \brief Negative log-likelihood of the sequences from the observed user-item pairs. 
 *
 * \f{align}{
 * 		- \frac{1}{|\Omega|}\sum_{\mathcal{T}^{u,i}}\bigg\{\sum_{t^{u,i}_j\in\mathcal{T}^{u,i}}\log\bigg(\boldsymbol{\Lambda}_0(u,i) + \boldsymbol{A}(u,i)\sum_{t^{u,i}_k < t^{u,i}_j}\exp(-\beta_{u,i}(t^{u,i}_j - t^{u,i}_k))\bigg) - T^{u,i}\boldsymbol{\Lambda}_0(u,i) - \boldsymbol{A}(u,i)\sum_{t^{u,i}_k < T^{u,i}}\int_{t^{u,i}_k}^{T^{u,i}}\lambda^{u,i}(t)dt\bigg\},
 * \f}
 * where \f$\mathcal{T}^{u,i}\f$ is the sequence of events induced from the pair \f$(u,i)\f$.
 * @param[out] objvalue negative loglikelihood. 
 * @param[out] gradient gradient of the parameters.
 */
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

	double PredictNextEventTime(unsigned uid, unsigned itemid, double T, const std::vector<Sequence>& data);

	unsigned PredictNextItem(unsigned uid, double t, const std::vector<Sequence>& data);

};

#endif