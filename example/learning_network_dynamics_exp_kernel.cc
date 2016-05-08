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

	std::cout << "2. Loading graph" << std::endl;
	Graph G("/Users/nandu/Development/exp_fit_graph/example_cascade_exp_1000_network", N, false);
	PlainTerminating terminating(num_params, dim, &G);	
	
	PlainTerminating::OPTION options;
	options.method = PlainTerminating::PLBFGS;
	options.excitation_regularizer = PlainTerminating::NONE;
	options.coefficients[PlainTerminating::LAMBDA] = 0;

	std::cout << "3. Fitting parameters" << std::endl << std::endl;
	terminating.fit(data, options);

	Eigen::VectorXd result = terminating.GetParameters();
	Eigen::Map<Eigen::MatrixXd> alpha_matrix = Eigen::Map<Eigen::MatrixXd>(result.data(), dim, dim);
	std::cout << std::endl << "Estimated Parameters : " << std::endl << alpha_matrix << std::endl;

	return 0;
}