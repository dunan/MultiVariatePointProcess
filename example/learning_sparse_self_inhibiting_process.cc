#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "SelfInhibitingProcess.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 6, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);
	params << 1.0, 1.0, 1.0, 1.5, 1.0, 1.4, 
			  0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
			  0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
			  0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
			  0.0, 0.1, 0.0, 0.1, 0.0, 0.0,
			  0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
			  0.0, 0.0, 0.1, 0.0, 0.1, 0.0;

	std::vector<Sequence> data;

	SelfInhibitingProcess inhibiting(num_params, dim);
	inhibiting.SetParameters(params);

	OgataThinning ot(dim);
	std::vector<double> vec_T(20, 10);
	ot.Simulate(inhibiting, vec_T, data);
	
	double T = 0;
	for(unsigned c = 0; c < data.size(); ++ c)
	{
		T = (T < data[c].GetTimeWindow() ? data[c].GetTimeWindow() : T);
	}

	SelfInhibitingProcess::OPTION options;
	options.base_intensity_regularizer = SelfInhibitingProcess::NONE;
	options.excitation_regularizer = SelfInhibitingProcess::L1;
	options.coefficients[SelfInhibitingProcess::LAMBDA0] = 0;
	options.coefficients[SelfInhibitingProcess::LAMBDA] = 1e1;

	SelfInhibitingProcess inhibiting_new(num_params, dim);

	inhibiting_new.fit(data, options);

	Eigen::VectorXd parameters_estimated = inhibiting_new.GetParameters();

	Eigen::Map<Eigen::VectorXd> Lambda0_hat = Eigen::Map<Eigen::VectorXd>(parameters_estimated.segment(0, dim).data(), dim);
	
	Eigen::Map<Eigen::MatrixXd> B_hat = Eigen::Map<Eigen::MatrixXd>(parameters_estimated.segment(dim, dim * dim).data(), dim, dim);
	
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << Lambda0_hat.transpose() << std::endl << std::endl;
	std::cout << B_hat << std::endl << std::endl;

	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	
	Eigen::Map<Eigen::MatrixXd> B = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);

	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << Lambda0.transpose() << std::endl << std::endl;
	std::cout << B << std::endl << std::endl;

	return 0;
}