#ifndef PROCESS_H
#define PROCESS_H

#include <vector>
#include "Sequence.h"

class IProcess
{

private:

	std::vector<double> parameters_;

	unsigned num_dims_;

public:

	IProcess(const unsigned& n, const unsigned& num_dims)
	{
		parameters_ = std::vector<double>(n, 0);

		num_dims_ = num_dims;
	}

	~IProcess(){}

	const std::vector<double>& GetParameters() {return parameters_;}

	unsigned GetNumDims(){return num_dims_;};

	void SetParameters(const std::vector<double>& v) 
	{
		parameters_ = v;
		num_dims_ = parameters_.size();
	}

	virtual void Initialize(const std::vector<Sequence>& data) = 0;

	virtual void NegLoglikelihood(double& objvalue, std::vector<double>& Gradient) = 0;

	virtual double Intensity(const double& t, const Sequence& data, std::vector<double>& intensity_dim) = 0;

	virtual double IntensityUpperBound(const double& t, const Sequence& data, std::vector<double>& intensity_upper_dim) = 0;

};

#endif