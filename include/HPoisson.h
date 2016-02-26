#ifndef HPROCESS_H
#define HPROCESS_H
#include <vector>
#include "Process.h"

class HPoisson : public IProcess
{

protected:

	std::vector<double> intensity_features_;
	
	double intensity_itegral_features_;

	unsigned num_sequences_;

public:

	HPoisson(const unsigned& n, const unsigned& num_dims) : IProcess(n, num_dims), num_sequences_(0) {}

	virtual void Initialize(const std::vector<Sequence>& data);

	virtual void NegLoglikelihood(double& objvalue, std::vector<double>& Gradient);

	virtual double Intensity(const double& t, const Sequence& data, std::vector<double>& intensity_dim);

	virtual double IntensityUpperBound(const double& t, const Sequence& data, std::vector<double>& intensity_upper_dim);

};

#endif