#include "PlainHawkes.h"
#include "SimpleRNG.h"

/*
	
	This class defines the SGD Hawkes process learner which privately inherits the PlainHawkes class to implement the contains relation.
	 
*/

class HawkesSGDLearner : private PlainHawkes
{

private:

	void StochasticGradient(const unsigned &k, std::vector<double>& gradient);

//  Initial learning rate;
	double ini_gamma_;

//  Maximum number of epochs;
	unsigned ini_max_iter_;

//  Internal implementation for random number generator;
	SimpleRNG RNG_;

public:

//  Constructor : n is the number of parameters; num_dims is the number of dimensions;
	HawkesSGDLearner(const unsigned& n, const unsigned& num_dims, const std::vector<double>& beta) : PlainHawkes(n, num_dims, beta), ini_gamma_(1e-2), ini_max_iter_(100) 
	{
		RNG_.SetState(0, 0);
	}
	
//	Fit the homogeneous poisson process to the given data and return the learned parameters in the variable returned_params; 
	void fit(const std::vector<Sequence>& data, std::vector<double>& returned_params);

};