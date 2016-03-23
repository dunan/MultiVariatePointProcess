#ifndef LOW_RANK_HAWKES_PROCESS_H
#define LOW_RANK_HAWKES_PROCESS_H
#include <vector>
#include <string>
#include <map>
#include "Process.h"
#include "Optimizer.h"

/*
	
	This class defines the Low Rank Hawkes process which implements the general process internface IPorcess.
	 
*/

class LowRankHawkesProcess : public IProcess
{

protected:

	Eigen::VectorXd beta_;

	Eigen::VectorXd event_intensity_features_;

	Eigen::VectorXd integral_intensity_features_;

	Eigen::SparseMatrix<double> pair_event_map_;

	Eigen::VectorXd observation_window_T_;

	Eigen::VectorXi observed_idx_;

	unsigned num_rows_;

	unsigned num_cols_;

//  This function requires process-specific implementation. It initializes the temporal features used to calculate the negative loglikelihood and the gradient. 
	void Initialize(const std::vector<Sequence>& data);

	unsigned Vec2Ind(const unsigned& i, const unsigned& j);

	void Ind2Vec(const unsigned& ind, unsigned& i, unsigned& j);

public:

	enum RegCoef {LAMBDA0, LAMBDA};

	//  Records the options

	struct OPTION
	{
		std::map<RegCoef, double> coefficients;	
	};

protected:

	LowRankHawkesProcess::OPTION options_;

public:

	LowRankHawkesProcess(const unsigned& num_rows, const unsigned& num_cols, const Eigen::VectorXd& beta) : IProcess(2 * num_rows * num_cols, num_rows * num_cols), beta_(beta), num_rows_(num_rows), num_cols_(num_cols)
	{
		options_.coefficients[LowRankHawkesProcess::LAMBDA0] = 0;
		options_.coefficients[LowRankHawkesProcess::LAMBDA] = 0;	
	}

//  MLE esitmation of the parameters
	void fit(const std::vector<Sequence>& data, const OPTION& options);

	void debugfit(const std::vector<Sequence>& data, const LowRankHawkesProcess::OPTION& options, const Eigen::MatrixXd& TrueLambda0, const Eigen::MatrixXd& TrueAlpha, const Eigen::VectorXd& X0);

//  This virtual function requires process-specific implementation. It calculates the negative loglikelihood of the given data. This function must be called after the Initialize method to return the negative loglikelihood of the data with respect to the current parameters. 
//	The returned negative loglikelihood is stored in the variable objvalue;
//	The returned gradient vector wrt the current parameters is stored in the variable Gradient; 
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	//  Return the stochastic gradient on the random sample k.
	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
	virtual double IntensityUpperBound(const double& t, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the integral of the intensity from a to b
	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

//  This function predicts the next event by simulation;
	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

};

#endif