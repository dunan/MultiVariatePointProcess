/**
 * \file Poisson.h
 * \brief The class definition of Poisson implementing the homogeneous Poisson process.
 */
#ifndef HPROCESS_H
#define HPROCESS_H
#include <vector>
#include "Process.h"


/**
 * \class Poisson Poisson.h "include/Poisson.h"
 * \brief Poisson implements the multivariate homogeneous process.
 *
 */

class Poisson : public IProcess
{

protected:

/**
 * \brief A column vector where each component is the average number of events on each dimension.
 */
	Eigen::VectorXd intensity_features_;

/**
 * \brief The average observation windown.
 */
	double intensity_itegral_features_;

/**
 * \brief total number of observed sequences
 */
	unsigned num_sequences_;

/**
 * \brief initialize the temporal features intensity_features_ and intensity_itegral_features_ from the input sequences.
 * @param[in] data input collection of sequences
 */
	void Initialize(const std::vector<Sequence>& data);

/**
 * \brief a column vector of length \f$C\f$ which is the total number of sequences. Each component records the observation window in the respective sequence.
 */
	Eigen::VectorXd observation_window_T_;

public:

	/**
	 * \brief The constructor 
	 * @param[in] n the number of parameters in total.
	 * @param[in] num_dims the number of dimensions in the process.
	 */
	Poisson(const unsigned& n, const unsigned& num_dims) : IProcess(n, num_dims), num_sequences_(0) {}

	/**
	 * \brief Negative loglikelihood of homogeneous Poisson process.
	 *
	 * \f{align}{
	 * -\frac{1}{C}\sum_{c=1}^C\bigg\{\sum_{n=1}^D (\sum_{i=1}^{n_c}\log\lambda^0_n - T_c\lambda^0_n) \bigg\}
	 * \f}
	 * @param objvalue negative loglikelihood. 
	 * @param gradient gradient of the parameters.
	 */
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& Gradient);

	/**
	 * \brief The intensity of each dimension for a homogeneous Poisson process is a constant \f$\lambda^0_n\f$.
	 * @param[in]  t         the given time.
	 * @param[in]  data      the given sequence of the past events until time t.
	 * @param[out]  intensity_dim a column vector of size num_dims_ where each component stores the intensity value of the respetive dimension at time t given the past sequence in data.
	 * @return               the summation of the intensity value from each dimension. 
	 */
	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

	/**
 	 * \brief Maximum likelihood estimation for the model parameters.
 	 * @param[in] data    vectors of observed sequences.
 	 */
	void fit(const std::vector<Sequence>& data);

};

#endif