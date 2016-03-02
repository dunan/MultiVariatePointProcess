#include <iostream>
#include "../include/TestModule.h"

void TestModule::TestHPoisson()
{
		// Test general Ogata Thinning algorithm
	// Set dimension = 2

	unsigned dim = 2;

	OgataThinning ot(dim);

	HPoisson hpoisson(dim,dim);
	std::vector<double> params;
	params.push_back(1.0);
	params.push_back(0.5);
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
	BasicPoissonLearner bpl(dim, dim);
	// Store estimated parameter vector
	std::vector<double> params_hat;
	bpl.fit(sequences, params_hat);
	// Print estimated parameter
	std::cout << "estimated : " << std::endl;
	for(unsigned d = 0; d < dim; ++ d)
	{
		std::cout << params_hat[d] << " ";
	}
	std::cout << std::endl;

	std::cout << "true : " << std::endl;
	for(unsigned d = 0; d < dim; ++ d)
	{
		std::cout << params[d] << " ";
	}
	std::cout << std::endl;
}

void TestModule::TestPlainHawkes()
{
	unsigned dim = 1;

	OgataThinning ot(dim);

	std::vector<double> params;
	params.push_back(0.1);
	params.push_back(0.5);

	std::vector<double> beta;
	beta.push_back(1.0);

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
	for(unsigned c = 0; c < sequences.size(); ++ c)
	{
		const std::vector<Event>& seq = sequences[c].GetEvents();
		for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
		{
			std::cout << i_event -> time << " " << i_event -> DimentionID << "; ";
		}
		std::cout << std::endl;
	}

	PlainHawkes hawkes_new(2, 1, beta);

	hawkes_new.fit(sequences, "SGD");

	const std::vector<double>& params_hat = hawkes_new.GetParameters();

	std::cout << "estimated : " << std::endl;
	for(unsigned d = 0; d < 2; ++ d)
	{
		std::cout << params_hat[d] << " ";
	}
	std::cout << std::endl;

	std::cout << "true : " << std::endl;
	for(unsigned d = 0; d < 2; ++ d)
	{
		std::cout << params[d] << " ";
	}
	std::cout << std::endl;


}

