#include "HPoisson.h"

class BasicPoissonLearner : private HPoisson
{
public:

	BasicPoissonLearner(const unsigned& n, const unsigned& num_dims) : HPoisson(n, num_dims) {}
	
	void fit(const std::vector<Sequence>& data, std::vector<double>& returned_params);

};