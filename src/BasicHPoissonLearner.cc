#include <iostream>

#include "../include/BasicPoissonLearner.h"

void BasicPoissonLearner::fit(const std::vector<Sequence>& data, std::vector<double>& returned_params)
{
	HPoisson::Initialize(data);

	const unsigned& D = IProcess::GetNumDims();

	returned_params = std::vector<double> (D, 0);
	
	for(unsigned d = 0; d < D; ++ d)
	{
		returned_params[d] = intensity_features_[d] / intensity_itegral_features_;
	}

}