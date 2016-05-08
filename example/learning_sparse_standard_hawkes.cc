#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"

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
	
	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	std::vector<Sequence> data;
	OgataThinning ot(dim);

	unsigned num_events = 1000, num_sequences = 100;
	std::cout << "1. Simulating " << num_sequences << " sequences with " << num_events << " events each " << std::endl;
	ot.Simulate(hawkes, num_events, num_sequences, data);

	for(unsigned c = 0; c < data.size(); ++ c)
	{
		std::map<unsigned, unsigned> dim2count;
		const std::vector<Event>& seq = data[c].GetEvents();
		for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
		{
			if(dim2count.find(i_event -> DimentionID) == dim2count.end())
			{
				dim2count[i_event -> DimentionID] = 1;
			}else
			{
				++ dim2count[i_event -> DimentionID];
			}
		}

		for(std::map<unsigned, unsigned>::const_iterator m = dim2count.begin(); m != dim2count.end(); ++ m)
		{
			std::cout << m->first << " " << m->second << std::endl;
		}

		std::cout << std::endl;
	}

	std::cout << "2. Finish simulating " << std::endl;
	
	PlainHawkes hawkes_new(num_params, dim, beta);
	
	PlainHawkes::OPTION options;
	options.method = PlainHawkes::PLBFGS;
	options.base_intensity_regularizer = PlainHawkes::L22;
	options.excitation_regularizer = PlainHawkes::L1;
	options.coefficients[PlainHawkes::LAMBDA] = 500;
	options.coefficients[PlainHawkes::BETA] = 70;

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