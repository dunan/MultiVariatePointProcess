#include <iostream>
#include <iomanip> 
#include <chrono>
#include <Eigen/Dense>
#include "Sequence.h"
#include "Utility.h"
#include "TerminatingProcessLearningTriggeringKernel.h"

int main(const int argc, const char** argv)
{
	std::vector<Sequence> data;
	unsigned N = 6;
	
	double T;
	ImportFromExistingCascades("data/example_network_cascade_general_kernel", N, T, data);

	std::cout << "1. Loaded " << data.size() << " sequences" << std::endl;

	unsigned dim = N, num_basis = 100, num_params = num_basis * dim * dim;

	Eigen::VectorXd tau = Eigen::VectorXd::LinSpaced(num_basis, 0, T);
	Eigen::VectorXd sigma = Eigen::VectorXd::Constant(tau.size(), 1.0);

	TerminatingProcessLearningTriggeringKernel terminating(num_params, dim, tau, sigma);
	TerminatingProcessLearningTriggeringKernel::OPTION options;
	options.excitation_regularizer = TerminatingProcessLearningTriggeringKernel::GROUP;
	options.coefficients[TerminatingProcessLearningTriggeringKernel::LAMBDA] = 1;

	std::cout << "2. Fitting parameters" << std::endl << std::endl;
	terminating.fit(data, options);

	std::cout << "3. Plotting learned triggering kernels" << std::endl << std::endl;
	terminating.PlotTriggeringKernel(0, 1, T, 0.01);

	return 0;
}