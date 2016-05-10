#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 5, num_params = dim * (dim + 1);

/**
 * Generate a 5-by-5 matrix with rank 2.
 */
	Eigen::MatrixXd B1 = (Eigen::MatrixXd::Random(dim,2).array() + 1) / 2;
	Eigen::MatrixXd B2 = (Eigen::MatrixXd::Random(dim,2).array() + 1) / 2;

/**
 * Simply guarantee the stationary condition of the mulivariate Hawkes process.
 */
	Eigen::MatrixXd B = B1 * B2.transpose() / 8;

	Eigen::EigenSolver<Eigen::MatrixXd> es(B);

	OgataThinning ot(dim);

	Eigen::VectorXd params(num_params);
	
	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);

	Lambda0 = Eigen::VectorXd::Constant(dim, 0.1);
	A = B;

	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1.0);

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	std::vector<Sequence> sequences;

	unsigned n = 1000;
	unsigned num_sequences = 10;
	ot.Simulate(hawkes, n, num_sequences, sequences);

	PlainHawkes hawkes_new(num_params, dim, beta);
	
	PlainHawkes::OPTION options;
	options.base_intensity_regularizer = PlainHawkes::NONE;
	options.excitation_regularizer = PlainHawkes::NUCLEAR;
	options.coefficients[PlainHawkes::BETA] = 0.1;
	options.ini_learning_rate = 5e-5;
	options.rho = 1;
	options.ub_nuclear = 1;
	options.ini_max_iter = 1000;
	hawkes_new.fit(sequences, options, params);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	std::cout << "True Parameters : " << std::endl;
	std::cout << params.transpose() << std::endl;

	return 0;
}