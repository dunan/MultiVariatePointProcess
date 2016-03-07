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
	unsigned dim = 1;

	OgataThinning ot(dim);

	Eigen::VectorXd params(2);
	params << 0.1, 0.5;

	Eigen::MatrixXd beta(1,1);

	beta << 1.0;

	PlainHawkes hawkes(2, 1, beta);
	hawkes.SetParameters(params);

	// Store the simulated sequences
	std::vector<Sequence> sequences;

	// Simulate 10 events for each sequence
	unsigned n = 1000;
	// Simulate 2 sequences 
	unsigned num_sequences = 1;
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

	PlainHawkes hawkes_new(2, 1, beta);
	// hawkes_new.fit(sequences, "SGD");
	hawkes_new.fit(sequences, "LBFGS");
	
	std::cout << "estimated : " << std::endl;
	std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	std::cout << "true : " << std::endl;
	std::cout << params.transpose() << std::endl;


}

