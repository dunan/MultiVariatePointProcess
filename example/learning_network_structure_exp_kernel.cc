#include <iostream>
#include <iomanip> 
#include <chrono>
#include <Eigen/Dense>
#include "Sequence.h"
#include "Utility.h"
#include "PlainTerminating.h"
#include "Graph.h"

int main(const int argc, const char** argv)
{
	std::vector<Sequence> data;
	unsigned N = 6;
	double T = 0;

	ImportFromExistingCascades("data/example_cascade_exp_1000", N, T, data);

	std::cout << "1. Loaded " << data.size() << " sequences" << std::endl;

	unsigned dim = N, num_params = dim * dim;

	PlainTerminating terminating(num_params, dim);

	PlainTerminating::OPTION options;
	options.method = PlainTerminating::PLBFGS;
	options.excitation_regularizer = PlainTerminating::L1;
	options.coefficients[PlainTerminating::LAMBDA] = 1e-3;

	std::cout << "2. Fitting parameters" << std::endl << std::endl;
	terminating.fit(data, options);

	Eigen::VectorXd result = terminating.GetParameters();
	Eigen::Map<Eigen::MatrixXd> alpha_matrix = Eigen::Map<Eigen::MatrixXd>(result.data(), dim, dim);
	std::cout << std::endl << "Estimated Parameters : " << std::endl << alpha_matrix << std::endl << std::endl;

	return 0;
}