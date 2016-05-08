#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 2, num_params = dim * (dim + 1);

	OgataThinning ot(dim);

	Eigen::VectorXd params(num_params);
	params << 0.1, 0.2, 0.5, 0.5, 0.5, 0.5; 

	Eigen::MatrixXd beta(dim,dim);
	beta << 1, 1, 1, 1;

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	std::vector<Sequence> sequences;

	unsigned num_events = 1000, num_sequences = 10;
	std::cout << "1. Simulating " << num_sequences << " sequences with " << num_events << " events each " << std::endl;

	ot.Simulate(hawkes, num_events, num_sequences, sequences);

	PlainHawkes hawkes_new(num_params, dim, beta);
	PlainHawkes::OPTION options;
	options.method = PlainHawkes::PLBFGS;
	options.base_intensity_regularizer = PlainHawkes::NONE;
	options.excitation_regularizer = PlainHawkes::NONE;

	std::cout << "2. Fitting Parameters " << std::endl << std::endl;  
	hawkes_new.fit(sequences, options);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	std::cout << "True Parameters : " << std::endl;
	std::cout << params.transpose() << std::endl;

	return 0;
}