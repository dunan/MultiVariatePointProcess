#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "SelfInhibitingProcess.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 2;
	unsigned num_params = dim * (dim + 1);

	Eigen::VectorXd params(num_params);
	params << 1.2, 1, 0.1, 0.1, 0.1, 0.05;

	std::vector<Sequence> sequences;

	SelfInhibitingProcess inhibiting(num_params, dim);
	inhibiting.SetParameters(params);

	std::vector<double> vec_T(100, 10);

	OgataThinning ot(dim);
	ot.Simulate(inhibiting, vec_T, sequences);

	SelfInhibitingProcess::OPTION options;
	options.base_intensity_regularizer = SelfInhibitingProcess::NONE;
	options.excitation_regularizer = SelfInhibitingProcess::NONE;
	options.coefficients[SelfInhibitingProcess::LAMBDA0] = 0;
	options.coefficients[SelfInhibitingProcess::LAMBDA] = 0;

	SelfInhibitingProcess inhibiting_new(num_params, dim);

	inhibiting_new.fit(sequences, options);

	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << inhibiting_new.GetParameters().transpose() << std::endl;
	std::cout << "True Parameters : " << std::endl;
	std::cout << params.transpose() << std::endl;


	return 0;
}