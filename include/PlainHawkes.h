#ifndef HPROCESS_H
#define HPROCESS_H
#include <vector>
#include "Process.h"


/*
	
	This class defines the Hawkes process which implements the general process internface IPorcess.
	 
*/

class PlainHawkes : public IProcess
{

protected:

//  This variable is process-specific. It stores the temporal features associated with the intensity function of the multivariate hawkes process.
//  for each sequence c, given a pair of dimension n and m, and a point t^m_{c,i} on the dimension m, all_exp_kernel_recursive_sum_[c][m][n][i] stores the exponential sum of all the past events t^n_{c,j} < t^m_{c,i} on the dimension n in the sequence c
	std::vector<std::vector<std::vector<std::vector<double> > > > all_exp_kernel_recursive_sum_;

//	This variable is process-specific. It stores the temporal features associated with the integral intensity function of the multivariate hawkes process. intensity_itegral_features_[c][m][n] stores the summation \sum_{t^n_{c,i} < T_c} (1 - exp(-\beta^nm(T_c - t^n_{c,i})))
	std::vector<std::vector<std::vector<double> > > intensity_itegral_features_;

	std::vector<std::vector<std::vector<double> > > all_timestamp_per_dimension_;

//  This variable is process-specific. It records totoal number of sequences used to fit the process.
	unsigned num_sequences_;

	const std::vector<double>& beta_;

//  This function requires process-specific implementation. It initializes the temporal features used to calculate the negative loglikelihood and the gradient. 
	void Initialize(const std::vector<Sequence>& data);

	unsigned Idx(const unsigned& i, const unsigned& j);

public:

//  Constructor : n is the number of parameters in total; num_dims is the number of dimensions in the process;
	PlainHawkes(const unsigned& n, const unsigned& num_dims, const std::vector<double>& beta) : IProcess(n, num_dims), beta_(beta), num_sequences_(0) {}

//  This virtual function requires process-specific implementation. It calculates the negative loglikelihood of the given data. This function must be called after the Initialize method to return the negative loglikelihood of the data with respect to the current parameters. 
//	The returned negative loglikelihood is stored in the variable objvalue;
//	The returned gradient vector wrt the current parameters is stored in the variable Gradient; 
	virtual void NegLoglikelihood(double& objvalue, std::vector<double>& Gradient);

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
	virtual double Intensity(const double& t, const Sequence& data, std::vector<double>& intensity_dim);

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
	virtual double IntensityUpperBound(const double& t, const Sequence& data, std::vector<double>& intensity_upper_dim);

};

#endif