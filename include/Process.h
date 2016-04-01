#ifndef PROCESS_H
#define PROCESS_H

#include <Eigen/Dense>
#include "Sequence.h"

/*
	
	This class defines the interface for a general process.
	 
*/
class IProcess
{

protected:

//  Internal representation for the vector of parameters
	Eigen::VectorXd parameters_;

//  number of dimensions in the process;
	unsigned num_dims_;

	std::vector<std::vector<std::vector<double> > > all_timestamp_per_dimension_;

	void InitializeDimension(const std::vector<Sequence>& data);

public:

//  Constructor : n is the number of parameters in total; num_dims is the number of dimensions in the process;
	IProcess(const unsigned& n, const unsigned& num_dims)
	{
		parameters_ = Eigen::VectorXd::Zero(n);

		num_dims_ = num_dims;
	}

	~IProcess(){}

//  Return the vector of parameters;
	const Eigen::VectorXd& GetParameters() {return parameters_;}

//  Return the number of dimensions;
	unsigned GetNumDims(){return num_dims_;};

//  Set the vector of parameters. This is useful when we initialize the parameters for optimization;
	void SetParameters(const Eigen::VectorXd& v) 
	{
		parameters_ = v;
	}


//  This virtual function requires process-specific implementation. It calculates the negative loglikelihood of the given data. This function must be called after the Initialize method to return the negative loglikelihood of the data with respect to the current parameters. 
//	The returned negative loglikelihood is stored in the variable objvalue;
//	The returned gradient vector wrt the current parameters is stored in the variable Gradient; 
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& Gradient) = 0;

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim) = 0;

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim) = 0;

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the integral of the intensity from a to b
	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data) = 0;

//  Return the stochastic gradient on the random sample k.
	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient) = 0;

};

#endif