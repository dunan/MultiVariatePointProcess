#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"
#include "HawkesLearningTriggeringKernel.h"

int main(const int argc, const char** argv)
{
	std::vector<Sequence> data;
	
	unsigned dim = 6, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);
	params << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
			  0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
			  0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
			  0.8, 0.2, 0.0, 0.0, 0.0, 0.0,
			  0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
			  0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
			  0.0, 0.0, 0.5, 0.0, 0.5, 0.5;
	
	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1);
	
	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	OgataThinning ot(dim);
	ot.Simulate(hawkes, 1000, 50 , data);

	double T = 0;
	for(unsigned c = 0; c < data.size(); ++ c)
	{
		T = (T < data[c].GetTimeWindow() ? data[c].GetTimeWindow() : T);
	}
	
	unsigned num_basis = 100, num_params_new = dim + num_basis * dim * dim;

	Eigen::VectorXd tau(num_basis);
	tau = Eigen::VectorXd::LinSpaced(num_basis + 1, 0, T).segment(0, num_basis);
	
	Eigen::VectorXd sigma = Eigen::VectorXd::Constant(tau.size(), 2.0);
	Graph G("data/example_network_hawkes", dim, false);

	HawkesLearningTriggeringKernel hawkes_learning_kernel(num_params_new, dim, &G, tau, sigma);

	HawkesLearningTriggeringKernel::OPTION options;
	options.excitation_regularizer = HawkesLearningTriggeringKernel::L22;
	options.coefficients[HawkesLearningTriggeringKernel::LAMBDA0] = 0;
	options.coefficients[HawkesLearningTriggeringKernel::LAMBDA] = 1.5e2;

	hawkes_learning_kernel.fit(data, options);

	for(unsigned i = 0; i < dim; ++ i)
	{
		hawkes_learning_kernel.PlotTriggeringKernel(i,i,T,0.01);	
	}
	return 0;
}