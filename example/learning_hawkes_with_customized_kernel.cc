#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "HawkesGeneralKernel.h"
#include "RayleighKernel.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 2, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);
	params << 0.1, 0.2, 0.5, 0.5, 0.5, 0.5; 
	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1);

	std::vector<std::vector<TriggeringKernel*> > triggeringkernels(dim, std::vector<TriggeringKernel*>(dim, NULL));

	for(unsigned m = 0; m < dim; ++ m)
	{
		for(unsigned n = 0; n < dim; ++ n)
		{
			triggeringkernels[m][n] = new RayleighKernel(1.0);
		}
	}

	HawkesGeneralKernel hawkes(num_params, dim, triggeringkernels);
	hawkes.SetParameters(params);
	OgataThinning ot(dim);
	std::vector<Sequence> sequences;
	ot.Simulate(hawkes, 1000, 100, sequences);

	HawkesGeneralKernel hawkes_new(num_params, dim, triggeringkernels);

	HawkesGeneralKernel::OPTION options;
	options.method = HawkesGeneralKernel::PLBFGS;
	options.base_intensity_regularizer = HawkesGeneralKernel::NONE;
	options.excitation_regularizer = HawkesGeneralKernel::NONE;

	hawkes_new.fit(sequences, options);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	std::cout << "True Parameters : " << std::endl;
	std::cout << params.transpose() << std::endl;

	return 0;
}