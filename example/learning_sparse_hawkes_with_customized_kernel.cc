#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "HawkesGeneralKernel.h"
#include "RayleighKernel.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 6, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);
	params << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
			  0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
			  0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
			  0.8, 0.2, 0.0, 0.0, 0.0, 0.0,
			  0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
			  0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
			  0.0, 0.0, 0.5, 0.0, 0.5, 0.5;

	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);
	
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

	unsigned num_events = 1000, num_sequences = 100;
	std::cout << "1. Simulating " << num_sequences << " sequences with " << num_events << " events each " << std::endl;
	OgataThinning ot(dim);
	std::vector<Sequence> data;
	ot.Simulate(hawkes, num_events, num_sequences, data);

	std::cout << "2. Finish simulating " << std::endl;
	
	HawkesGeneralKernel hawkes_new(num_params, dim, triggeringkernels);
	
	HawkesGeneralKernel::OPTION options;
	options.method = HawkesGeneralKernel::PLBFGS;
	options.base_intensity_regularizer = HawkesGeneralKernel::L22;
	options.excitation_regularizer = HawkesGeneralKernel::L1;
	options.coefficients[HawkesGeneralKernel::LAMBDA0] = 500;
	options.coefficients[HawkesGeneralKernel::LAMBDA] = 70;

	std::cout << "3. Fitting Parameters " << std::endl;  
	hawkes_new.fit(data, options);

	Eigen::VectorXd parameters_estimated = hawkes_new.GetParameters();

	Eigen::Map<Eigen::VectorXd> Lambda0_hat = Eigen::Map<Eigen::VectorXd>(parameters_estimated.segment(0, dim).data(), dim);
	
	Eigen::Map<Eigen::MatrixXd> A_hat = Eigen::Map<Eigen::MatrixXd>(parameters_estimated.segment(dim, dim * dim).data(), dim, dim);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << Lambda0_hat.transpose() << std::endl << std::endl;
	std::cout << A_hat << std::endl << std::endl;

	std::cout << "True Parameters " << std::endl;
	std::cout << Lambda0.transpose() << std::endl << std::endl;
	std::cout << A << std::endl << std::endl;


	return 0;
}