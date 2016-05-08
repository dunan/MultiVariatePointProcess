/**
 * \file SelfInhibitingProcess.h
 * \brief The class definition of SelfInhibitingProcess implementing the standard Self-Inhibiting (or Self-Correcting) process.
 */
#ifndef SELF_INHIBITING_PROCESS
#define SELF_INHIBITING_PROCESS
#include <vector>
#include <cstdlib>
#include <map>
#include "Process.h"
#include "Optimizer.h"

/**
 * \class SelfInhibitingProcess SelfInhibitingProcess.h "include/SelfInhibitingProcess.h"
 * \brief SelfInhibitingProcess implements the standard multivariate Self-inhibiting (or self-correcting) process.
 *
 * In a multivariate Self-inhibiting process, the intensity of each dimension \f$n\f$ is defined as:
 * \f{align}{
 * \lambda^*_n(t) = exp\bigg(\lambda^n_0t - \sum_{m=1}^D\sum_{t^m_j < t}\beta_{mn}\bigg),
 * \f}
 * where \f$\{\lambda^n_0\}\f$ and \f$\{\beta_{mn}\}\f$ are the model parameters. In contrast to the Hawkes process, the intuition here is that while the intensity increases steadily with the rate \f$\lambda^n_0\f$, every time when a new event appears, it is decreased by multiplying a constant $e^{-\beta_{mn}} < 1$, so the chance of new points decreases after an event has occurred recently. 
 */
class SelfInhibitingProcess : public IProcess
{

protected:

/**
 * \brief Temporal features associated with the intensity function.
 *
 * arrayK_[c][n] is an \f$n_c\f$ by \f$D\f$ matrix where arrayK_[c][n][i][m] records how many events have occurred on dimension \f$m\f$ before the \f$i\f$-th event on dimension \f$n\f$ in the given sequence \f$c\f$.
 */
	std::vector<std::vector<Eigen::MatrixXd > > arrayK_;

/**
 * \brief Temporal features associated with the integral of the intensity function.
 *
 * arrayG_[c][n][i] is a vector of pairs. Each pair records how many events have occurred on each dimension between the \f$(i-1)\f$-th event and the \f$i\f$-th event on the dimension \f$n\f$.
 */
	std::vector<std::vector<std::vector<std::vector<std::pair<double, unsigned> > > > > arrayG_;

/**
 * \brief Total number of observed sequences
 */
	unsigned num_sequences_;

/**
 * \brief A column vector of length \f$C\f$ which is the total number of sequences. Each component records the observation window in the respective sequence.
 */
	Eigen::VectorXd observation_window_T_;

/**
 * \brief Initialize the temporal features [arrayK](@ref arrayK) and [arrayG](@ref arrayG) from the input sequences
 * @param[in] data input collection of sequences
 */
	void Initialize(const std::vector<Sequence>& data);

/**
 * \brief restore to the default optimization configuration
 */
	void RestoreOptionToDefault();

/**
 * \brief Negative loglikelihood of self-inhibiting point process.
 *
 * \f{align}{
 * -\frac{1}{C}\sum_{c=1}^C\bigg\{\sum_{n = 1}^D\bigg(\sum_{i=1}^{n_c}\bigg(\lambda^n_0t^n_{i,c} - \sum_{m=1}^D\sum_{t^m_{j,c} < t^n_{i,c}}\beta_{mn} - \int_{t^n_{i-1,c}}^{t^n_{i,c}}\exp(\lambda^n_0t - \sum_{m=1}^D\sum_{t^m_{j,c} < t}\beta_{mn})dt\bigg) - \int_{t^n_{n_c,c}}^{T_c}\exp(\lambda^n_0t - \sum_{m=1}^D\sum_{t^m_{j,c} < t}\beta_{mn})dt\bigg)\bigg\}
 * \f}
 * @param[out] objvalue negative loglikelihood. 
 * @param[out] gradient gradient of the parameters.
 */
	void GetNegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);
/**
 * \brief Post process the learned dependency structure
 */
	void PostProcessing();

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
		 * Regularization coefficient for \f$\|\boldsymbol{\beta}\|\f$
		 */
		LAMBDA
	};

	/**
	 * \brief Options used to configure the fitting of the terminating point process.
	 */
	struct OPTION
	{
		/**
		 * Type of regularization used for \f$\|\boldsymbol{\lambda}_0\|\f$
		 */
		Regularizer base_intensity_regularizer;
		/**
		 * Type of regularization used for \f$\|\boldsymbol{\beta}\|\f$
		 */
		Regularizer excitation_regularizer;
		/**
		 * Regularization coefficients value
		 */
		std::map<RegCoef, double> coefficients;	
	};

protected:

	OPTION options_;

public:

	/**
	 * \brief The constructor 
	 * @param[in] n the number of parameters in total.
	 * @param[in] num_dims the number of dimensions in the process.
	 */
	SelfInhibitingProcess(const unsigned& n, const unsigned& num_dims) : IProcess(n, num_dims), num_sequences_(0)
	{
		options_.base_intensity_regularizer = NONE;
		options_.excitation_regularizer = NONE;
		options_.coefficients[LAMBDA0] = 0;
		options_.coefficients[LAMBDA] = 0;
	}

	/**
 	 * \brief Maximum likelihood estimation for the model parameters.
 	 * @param[in] data    vectors of observed sequences.
 	 * @param[in] options data structure sotring different configuration for the optimization algorithm and the respective regularizations.
 	 */
	void fit(const std::vector<Sequence>& data, const OPTION& options);

/**
 * \brief Negative loglikelihood of self-inhibiting point process.
 *
 * Call [GetNegLoglikelihood](@ref GetNegLoglikelihood) with different types of regularization.
 * @param[out] objvalue negative loglikelihood. 
 * @param[out] gradient gradient of the parameters.
 */
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);
/**
 * \brief Intensity function of self-inhibiting point process.
 *
 * In a multivariate Self-inhibiting process, the intensity of each dimension \f$n\f$ is defined as:
 * \f{align}{
 * \lambda^*_n(t) = exp\bigg(\lambda^n_0t - \sum_{m=1}^D\sum_{t^m_j < t}\beta_{mn}\bigg),
 * \f}
 * where \f$\{\lambda^n_0\}\f$ and \f$\{\beta_{mn}\}\f$ are the model parameters. In contrast to the Hawkes process, the intuition here is that while the intensity increases steadily with the rate \f$\lambda^n_0\f$, every time when a new event appears, it is decreased by multiplying a constant $e^{-\beta_{mn}} < 1$, so the chance of new points decreases after an event has occurred recently. 
 * @param[in]  t         the given time.
 * @param[in]  data      the given sequence of the past events until time t.
 * @param[out] intensity_dim a column vector of size num_dims_ where each component stores the intensity value of the respetive dimension at time t given the sequence of past events in data.
 * @return               the summation of the intensity value from each dimension. 
 */
	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);
	
};

#endif