#include <iostream>
#include "../include/TestModule.h"

void TestModule::TestHPoisson()
{
		// Test general Ogata Thinning algorithm
	// Set dimension = 2

	unsigned dim = 2;

	OgataThinning ot(dim);

	HPoisson hpoisson(dim,dim);

	Eigen::VectorXd params(dim);
	params << 1.0, 0.5;
	hpoisson.SetParameters(params);

	// Store the simulated sequences
	std::vector<Sequence> sequences;

	// Simulate 10 events for each sequence
	unsigned n = 10;
	// Simulate 2 sequences 
	unsigned num_sequences = 2;
	ot.Simulate(hpoisson, n, num_sequences, sequences);
	// Print simulated sequences
	for(unsigned c = 0; c < sequences.size(); ++ c)
	{
		const std::vector<Event>& seq = sequences[c].GetEvents();
		for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
		{
			std::cout << i_event -> time << " " << i_event -> DimentionID << "; ";
		}
		std::cout << std::endl;
	}

	// Learn the parameters back from the simulated sequences
	// BasicPoissonLearner bpl(dim, dim);
	// Store estimated parameter vector
	// Eigen::VectorXd params_hat;
	// bpl.fit(sequences, params_hat);
	// Print estimated parameter
	HPoisson hpoisson1(dim,dim);
	hpoisson1.fit(sequences);
	std::cout << "estimated : " << std::endl;
	std::cout << hpoisson1.GetParameters().transpose() << std::endl;

	std::cout << "true : " << std::endl;
	std::cout << params.transpose() << std::endl;
}

void TestModule::TestPlainHawkes()
{
	unsigned dim = 2, num_params = dim * (dim + 1);

	OgataThinning ot(dim);

	Eigen::VectorXd params(num_params);
	params << 0.1, 0.2, 0.5, 0.5, 0.5, 0.5; 
			  
	std::cout << params << std::endl;

	Eigen::MatrixXd beta(dim,dim);
	beta << 1, 1, 1, 1;
	std::cout << beta << std::endl;

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	// Store the simulated sequences
	std::vector<Sequence> sequences;

	// Simulate 10 events for each sequence
	unsigned n = 2000;
	// Simulate 2 sequences 
	unsigned num_sequences = 10;
	ot.Simulate(hawkes, n, num_sequences, sequences);

	// Print simulated sequences
	// for(unsigned c = 0; c < sequences.size(); ++ c)
	// {
	// 	const std::vector<Event>& seq = sequences[c].GetEvents();
	// 	for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
	// 	{
	// 		std::cout << i_event -> time << " " << i_event -> DimentionID << "; ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// double avg = 0;
	// const std::vector<Event>& seq = sequences[0].GetEvents();
	// for(unsigned i = 0; i < 100; ++ i)
	// {
	// 	Event event = ot.SimulateNext(hawkes, sequences[0]);
	// 	std::cout << i << " " << event.time << " " << event.DimentionID << std::endl;	
	// 	avg += event.time;
	// }
	// std::cout << avg / 100 << std::endl;
	// std::cout << hawkes.PredictNextEventTime(sequences[0], 100) << std::endl;

	PlainHawkes hawkes_new(num_params, dim, beta);
	// hawkes_new.fit(sequences, "SGD");
	PlainHawkes::OPTION options;
	options.method = PLBFGS;
	options.base_intensity_regularizer = NONE;
	options.excitation_regularizer = NONE;

	hawkes_new.fit(sequences, options);
	
	std::cout << "estimated : " << std::endl;
	std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	std::cout << "true : " << std::endl;
	std::cout << params.transpose() << std::endl;

	dim = 1;
	num_params = dim * (dim + 1);

	Eigen::VectorXd params1(num_params);
	params1 << 0.1, 0.5;

	Eigen::MatrixXd beta1(dim,dim);
	beta1 << 1;

	PlainHawkes hawkes1(num_params, dim, beta1);
	hawkes1.SetParameters(params1);

	OgataThinning ot1(dim);
	sequences.clear();
	ot1.Simulate(hawkes1, 1000, 1, sequences);

	// for(unsigned c = 0; c < sequences.size(); ++ c)
	// {
	// 	const std::vector<Event>& seq = sequences[c].GetEvents();
	// 	for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
	// 	{
	// 		std::cout << i_event -> time << " " << i_event -> DimentionID << "; ";
	// 	}
	// 	std::cout << std::endl;
	// }

	std::cout << Diagnosis::TimeChangeFit(hawkes1, sequences[0]) << std::endl;
}


void TestModule::TestMultivariateTerminating()
{
	std::vector<Sequence> data;
	unsigned N = 6;
	// double T = 8.2315;
	double T = 0;
	ImportFromExistingCascades("/Users/nandu/Development/exp_fit_graph/example_cascade_exp_10", N, T, data);

	unsigned dim = N, num_params = dim * dim;

	// Eigen::VectorXd alpha = Eigen::VectorXd::Constant(dim * dim, 1);
	PlainTerminating terminating(num_params, dim);

	// Graph G("/Users/nandu/Development/exp_fit_graph/example_cascade_exp_10_network", 6);
	// G.LoadWeibullFormatNetwork(",", false);
	// G.PrintWblNetwork();
	// PlainTerminating terminating(num_params, dim, &G);	
	
	PlainTerminating::OPTION options;
	options.method = PlainTerminating::PLBFGS;
	options.excitation_regularizer = PlainTerminating::NONE;
	options.coefficients[PlainTerminating::LAMBDA] = 0;

	terminating.fit(data, options);

	Eigen::VectorXd result = terminating.GetParameters();

	Eigen::Map<Eigen::MatrixXd> alpha_matrix = Eigen::Map<Eigen::MatrixXd>(result.data(), dim, dim);

	std::cout << alpha_matrix << std::endl;

}

void TestModule::TestTerminatingProcessLearningTriggeringKernel()
{
	std::vector<Sequence> data;
	unsigned N = 6;
	// double T = 8.2315;
	double T = 0;
	ImportFromExistingCascades("/Users/nandu/Development/exp_fit_graph/example_cascade_exp_10", N, T, data);

	unsigned dim = N, num_basis = 50, num_params = num_basis * dim * dim;

	Eigen::VectorXd tau = Eigen::VectorXd::LinSpaced(num_basis, 0, 10);
	Eigen::VectorXd sigma = Eigen::VectorXd::Constant(tau.size(), 0.5);

	std::cout << tau.transpose() << std::endl;
	std::cout << sigma.transpose() << std::endl;

	// Graph G("/Users/nandu/Development/exp_fit_graph/example_cascade_exp_10_network", N);
	// G.LoadWeibullFormatNetwork(",", false);
	// G.PrintWblNetwork();

	// TerminatingProcessLearningTriggeringKernel terminating(num_params, dim, &G, tau, sigma);
	// TerminatingProcessLearningTriggeringKernel::OPTION options;
	// options.excitation_regularizer = TerminatingProcessLearningTriggeringKernel::L22;
	// options.coefficients[TerminatingProcessLearningTriggeringKernel::LAMBDA] = 0.5;

	TerminatingProcessLearningTriggeringKernel terminating(num_params, dim, tau, sigma);
	TerminatingProcessLearningTriggeringKernel::OPTION options;
	options.excitation_regularizer = TerminatingProcessLearningTriggeringKernel::GROUP;
	options.coefficients[TerminatingProcessLearningTriggeringKernel::LAMBDA] = 1;

	terminating.fit(data, options);

}
