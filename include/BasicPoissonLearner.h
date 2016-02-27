#include "HPoisson.h"

/*
	
	This class defines the Basic Homogeneous Poisson learner which privately inherits the HPoisson class to implement the contains relation.
	 
*/

class BasicPoissonLearner : private HPoisson
{
public:

//  Constructor : n is the number of parameters; num_dims is the number of dimensions;
	BasicPoissonLearner(const unsigned& n, const unsigned& num_dims) : HPoisson(n, num_dims) {}
	
//	Fit the homogeneous poisson process to the given data and return the learned parameters in the variable returned_params; 
	void fit(const std::vector<Sequence>& data, std::vector<double>& returned_params);

};