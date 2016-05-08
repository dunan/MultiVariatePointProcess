/**
 * \file Process.h
 * \brief The class definition of Process which defines the general interface of a point process.
 */
#ifndef PROCESS_H
#define PROCESS_H

#include <Eigen/Dense>
#include "Sequence.h"


/**
 *	\interface IProcess Process.h "include/Process.h"
 *	
 *	\brief IProcess defines a general interface for each specific point process.
 *
 *  The IProcess class provides the basic interface for a general point process, which can be used for general fitting, simulation, and model checking. Customized point processes are supported by simply inheriting from this class. 
 *  
 */
class IProcess
{

	protected:

	/**
	 * 	\brief A column vector represents all model parameters of the process.
	 */
		Eigen::VectorXd parameters_;

	/**
	 * 	\brief The total number of dimensions of the process.
	 */
		unsigned num_dims_;

	/**
	 * 	\brief all_timestamp_per_dimension_ is a 3-d array where all_timestamp_per_dimension_[c][n][i] records the i-th event on the n-th dimension in the c-th sequence.
	 */
		std::vector<std::vector<std::vector<double> > > all_timestamp_per_dimension_;

	/**
	 * Assign each event of each sequence in the variable data to the respective dimesion by initializing the varaible all_timestamp_per_dimension_.
	 * @param[in] data collection of sequences.
	 */
		void InitializeDimension(const std::vector<Sequence>& data);

	public:

	/**
	 * 	\brief The constructor 
	 * 	@param[in] n the number of parameters in total.
	 * 	@param[in] num_dims the number of dimensions in the process.
	 *
	 */
		IProcess(const unsigned& n, const unsigned& num_dims)
		{
			parameters_ = Eigen::VectorXd::Zero(n);

			num_dims_ = num_dims;
		}

	/**
	 * \brief Return the column vector of model parameters.
	 * @return the column vector of model parameters.
	 */
		const Eigen::VectorXd& GetParameters() {return parameters_;}

	/**
	 * \brief Return the number of dimensions in the process.
	 * @return the number of dimensions in the process.
	 */
		unsigned GetNumDims(){return num_dims_;};

	/**
	 * \brief Set the model parameters. 
	 * @param[in] v A column vector storing the new values for the model parameters.
	 */
		void SetParameters(const Eigen::VectorXd& v) 
		{
			parameters_ = v;
		}

	/**
	 * \brief Returns the negative loglikelihood value and the gradient vectors.
	 * Given a collection of sequences \f$\mathcal{C} = \{\mathcal{S}^c\}\f$ where \f$\mathcal{S}^c = \{t^{c,d}_i\}^{d=1\dotso D}_{i=1\dotso n^c_d}\f$, and \f$n^c_d\f$ is the number of events on dimension d in sequence \f$\mathcal{S}^c\f$. The negative loglihood of observing \f$\mathcal{C}\f$ is defined as 
	 * \f{align}{
	 * \ell(\mathcal{C}) = - \sum_{d=1}^D\frac{1}{|\mathcal{C}|}\sum_{c=1}^{|\mathcal{C}|}\Bigg(\sum_{i=1}^{n^c_d}\lambda^*_d(t_i^{c,d}) - \int_{0}^{T_c}\lambda^*_d(\tau)d\tau\Bigg),
	 * \f}
	 * where \f$T_c\f$ is the observation window of the sequence \f$\mathcal{S}^c\f$.
	 * @param[out] objvalue the objective function value, which is the negative log-likelihood value evaluated on the given set of sequences using the current model parameters.
	 * @param[out] Gradient the gradient vector w.r.t. the model parameters.
	 */
		virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& Gradient) = 0;


	/**
	 * \brief Returns the summation \f$\sum_{d=1}^D\lambda^*_d(t)\f$ of the intensity value \f$\lambda^*_d(t)\f$ of each dimension in a given sequence data at the time t.
	 * @param[in]  t         the given time.
	 * @param[in]  data      the given sequence of the past events until time t.
	 * @param[out]  intensity_dim a column vector of size num_dims_ where each component stores the intensity value of the respetive dimension at time t given the past sequence in data.
	 * @return               the summation of the intensity value from each dimension. 
	 */
		virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim) = 0;

	/**
	 * \brief Returns the upper bound of the summation of the intensity value on each dimension from time t to t + L given the history of past events in sequence data.
	 * Let \f${\lambda_d^*(t)}\f$ be the conditional intensity function on the d-th dimension where \f$d=1\dotso D\f$, and num_dims_ = D. This function returns 
	 * \f{align}{
	 *	\lambda_0^D(t) \geq \sum_{d=1}^D\sup_{\tau\in[t, t + \tau(t)]}\lambda^*_d(\tau),
	 * \f}
	 * where the returned value \f$\lambda_0^D(t)\f$ will be used for Ogata's Thinning algorithm. 
	 * @param  t                   the starting time.
	 * @param  L                   the duration. 
	 * @param  data                the given sequence of the past events until time t.
	 * @param  intensity_upper_dim a column vector of size num_dims_ storing the upper bound of the intensity function on each dimension from time t to t + L.
	 * @return                     the summation of the upper-bound of each intensity function from the respetive dimension within the interval [t, t + L].
	 */
		virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim) = 0;

	/**
	 * \brief Returns the integral of the intensity function \f$\int_{a}^b\lambda^*(\tau)d\tau\f$ where \f$a = lower\f$ and \f$b = upper\f$.
	 * @param[in]  lower starting point of the integral.
	 * @param[in]  upper ending point of the integral.
	 * @param[in]  data  sequence of past events.
	 * @return       \f$\int_{a}^b\lambda^*(\tau)d\tau\f$ where \f$a = lower\f$ and \f$b = upper\f$.
	 */
		virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data) = 0;

	/**
	 * \brief Returns the gradient w.r.t. the model parameters on the k-th sequence.
	 * @param[in] k        sequence index.
	 * @param[out] gradient the gradient vector w.r.t. the model parameters.
	 */
		virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient) = 0;

	/**
	 * \brief Plots the intensity functions based on the given sequence. 
	 * It plots the intensity function and the associated event points up of each dimension in the same figure. 
	 * \warning Currently, it supports the ploting of up to four dimensions in total.
	 * @param[in] data an input sequence of events.
	 */
		void PlotIntensityFunction(const Sequence& data);

	/**
	 * \brief Plots the intensity function and the associated event points of the dimension dim_id.
	 * @param[in] data   an input sequence of events.
	 * @param[in] dim_id the index of the dimension we want to plot.
	 */
		void PlotIntensityFunction(const Sequence& data, const unsigned& dim_id);

	/**
 * \brief Predict the next event timing by the expectation \f$\int_{t_n}^\infty tf^*(t)dt\f$. Currently, we use the sample average by simulations to approximate the expectation since the conditional density \f$f^*(t)\f$ normally does not have an analytic form.
 * @param[in]  data            the sequence of past events.
 * @param[in]  num_simulations number of simulations we use to calculate the sample average.
 * @return                 the prediction of the next event timing.
 */
	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations) = 0;

};

#endif