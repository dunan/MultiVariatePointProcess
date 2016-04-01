#include <iostream>
#include <iomanip> 
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

	Eigen::MatrixXd beta(dim,dim);
	beta << 1, 1, 1, 1;

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	// Store the simulated sequences
	std::vector<Sequence> sequences;

	unsigned num_events = 1000, num_sequences = 10;
	std::cout << "1. Simulating " << num_sequences << " sequences with " << num_events << " events each " << std::endl;

	ot.Simulate(hawkes, num_events, num_sequences, sequences);

	PlainHawkes hawkes_new(num_params, dim, beta);
	// hawkes_new.fit(sequences, "SGD");
	PlainHawkes::OPTION options;
	options.method = PLBFGS;
	options.base_intensity_regularizer = NONE;
	options.excitation_regularizer = NONE;

	std::cout << "2. Fitting Parameters " << std::endl << std::endl;  
	hawkes_new.fit(sequences, options);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	std::cout << "True Parameters : " << std::endl;
	std::cout << params.transpose() << std::endl;

	// unsigned dim = 1;
	// unsigned num_params = dim * (dim + 1);

	// Eigen::VectorXd params1(num_params);
	// params1 << 0.1, 0.5;

	// Eigen::MatrixXd beta1(dim,dim);
	// beta1 << 1;

	// PlainHawkes hawkes1(num_params, dim, beta1);
	// hawkes1.SetParameters(params1);

	// std::vector<Sequence> sequences;
	// OgataThinning ot1(dim);
	// sequences.clear();
	// ot1.Simulate(hawkes1, 1000, 1, sequences);
	
	// for(unsigned c = 0; c < sequences.size(); ++ c)
	// {
	// 	const std::vector<Event>& seq = sequences[c].GetEvents();
	// 	for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
	// 	{
	// 		// std::cout << i_event -> time << " " << i_event -> DimentionID << "; ";
	// 		std::cout << std::setprecision(16) << i_event -> time << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << Diagnosis::TimeChangeFit(hawkes1, sequences[0]) << std::endl;
}

void TestModule::TestSparsePlainHawkes()
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
	
	// std::cout << A << std::endl;
	
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
	options.method = PLBFGS;
	options.base_intensity_regularizer = L22;
	options.excitation_regularizer = L1;
	options.coefficients[LAMBDA] = 500;
	options.coefficients[BETA] = 70;

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

	
}

void TestModule::TestMultivariateTerminating()
{
	std::vector<Sequence> data;
	unsigned N = 6;
	double T = 0;

	ImportFromExistingCascades("/Users/nandu/Development/exp_fit_graph/example_cascade_exp_1000", N, T, data);

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

}


void TestModule::TestMultivariateTerminatingWithUnknownStructure()
{
	std::vector<Sequence> data;
	unsigned N = 6;
	double T = 0;

	ImportFromExistingCascades("/Users/nandu/Development/exp_fit_graph/example_cascade_exp_1000", N, T, data);

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
	std::cout << std::endl << "Estimated Parameters : " << std::endl << alpha_matrix << std::endl;

}

void TestModule::TestTerminatingProcessLearningTriggeringKernel()
{
	std::vector<Sequence> data;
	unsigned N = 6;
	
	double T;
	ImportFromExistingCascades("../data/example_network_cascade", N, T, data);

	std::cout << "1. Loaded " << data.size() << " sequences" << std::endl;

	unsigned dim = N, num_basis = 100, num_params = num_basis * dim * dim;

	Eigen::VectorXd tau = Eigen::VectorXd::LinSpaced(num_basis, 0, T);
	Eigen::VectorXd sigma = Eigen::VectorXd::Constant(tau.size(), 1.0);

	std::cout << "2. Loading graph" << std::endl;
	Graph G("../data/example_network", N, false);

	TerminatingProcessLearningTriggeringKernel terminating(num_params, dim, &G, tau, sigma);
	TerminatingProcessLearningTriggeringKernel::OPTION options;
	options.excitation_regularizer = TerminatingProcessLearningTriggeringKernel::L22;
	options.coefficients[TerminatingProcessLearningTriggeringKernel::LAMBDA] = 1;

	std::cout << "3. Fitting parameters" << std::endl << std::endl;
	terminating.fit(data, options);

	std::cout << "4. Plotting learned triggering kernels" << std::endl;
	terminating.PlotTriggeringKernel(0, 1, T, 0.01);
}


void TestModule::TestTerminatingProcessLearningTriggeringKernelWithUnknownStructure()
{
	std::vector<Sequence> data;
	unsigned N = 6;
	
	double T;
	ImportFromExistingCascades("../data/example_network_cascade", N, T, data);

	std::cout << "1. Loaded " << data.size() << " sequences" << std::endl;

	unsigned dim = N, num_basis = 100, num_params = num_basis * dim * dim;

	Eigen::VectorXd tau = Eigen::VectorXd::LinSpaced(num_basis, 0, T);
	Eigen::VectorXd sigma = Eigen::VectorXd::Constant(tau.size(), 1.0);

	TerminatingProcessLearningTriggeringKernel terminating(num_params, dim, tau, sigma);
	TerminatingProcessLearningTriggeringKernel::OPTION options;
	options.excitation_regularizer = TerminatingProcessLearningTriggeringKernel::GROUP;
	options.coefficients[TerminatingProcessLearningTriggeringKernel::LAMBDA] = 1;

	std::cout << "2. Fitting parameters" << std::endl << std::endl;
	terminating.fit(data, options);

	std::cout << "3. Plotting learned triggering kernels" << std::endl << std::endl;
	terminating.PlotTriggeringKernel(0, 1, T, 0.01);

}

void TestModule::TestHawkesLearningTriggeringKernelUnknownStructure()
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

	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);
	
	std::cout << A << std::endl;
	
	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1);
	
	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	OgataThinning ot(dim);
	ot.Simulate(hawkes, 1000, 20, data);

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
	
	double T = 0;
	for(unsigned c = 0; c < data.size(); ++ c)
	{
		T = (T < data[c].GetTimeWindow() ? data[c].GetTimeWindow() : T);
	}

	std::cout << T << std::endl;
	
	unsigned num_basis = 200, num_params_new = dim + num_basis * dim * dim;

	Eigen::VectorXd tau(num_basis);
	tau = Eigen::VectorXd::LinSpaced(num_basis + 1, 0, T).segment(0, num_basis);
	std::cout << tau.transpose() << std::endl;
	Eigen::VectorXd sigma = Eigen::VectorXd::Constant(tau.size(), 1.0);

	HawkesLearningTriggeringKernel hawkes_learning_kernel(num_params_new, dim, tau, sigma);

	HawkesLearningTriggeringKernel::OPTION options;
	options.excitation_regularizer = HawkesLearningTriggeringKernel::GROUP;
	options.coefficients[HawkesLearningTriggeringKernel::LAMBDA0] = 0;
	options.coefficients[HawkesLearningTriggeringKernel::LAMBDA] = 5e2;

	hawkes_learning_kernel.fit(data, options);

	for(unsigned i = 0; i < dim; ++ i)
	{
		hawkes_learning_kernel.PlotTriggeringKernel(i,i,T,0.01);	
	}
}

void TestModule::TestHawkesLearningTriggeringKernel()
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

	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);
	
	std::cout << A << std::endl;
	
	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1);
	
	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	OgataThinning ot(dim);
	ot.Simulate(hawkes, 1000, 50 , data);

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
	
	double T = 0;
	for(unsigned c = 0; c < data.size(); ++ c)
	{
		T = (T < data[c].GetTimeWindow() ? data[c].GetTimeWindow() : T);
	}

	std::cout << T << std::endl;
	
	unsigned num_basis = 100, num_params_new = dim + num_basis * dim * dim;

	Eigen::VectorXd tau(num_basis);
	tau = Eigen::VectorXd::LinSpaced(num_basis + 1, 0, T).segment(0, num_basis);
	std::cout << tau.transpose() << std::endl;
	
	Eigen::VectorXd sigma = Eigen::VectorXd::Constant(tau.size(), 2.0);
	Graph G("../data/example_network_hawkes", dim, false);
	G.PrintWeibullFormatNetwork();

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
	// hawkes_learning_kernel.PlotTriggeringKernel(0,1,T,0.01);
	// hawkes_learning_kernel.PlotTriggeringKernel(0,2,T,0.01);
	// hawkes_learning_kernel.PlotTriggeringKernel(1,0,T,0.01);
	// hawkes_learning_kernel.PlotTriggeringKernel(1,2,T,0.01);
	// hawkes_learning_kernel.PlotTriggeringKernel(1,3,T,0.01);
	// hawkes_learning_kernel.PlotTriggeringKernel(2,5,T,0.01);
	// hawkes_learning_kernel.PlotTriggeringKernel(2,5,T,0.01);
	// hawkes_learning_kernel.PlotTriggeringKernel(4,5,T,0.01);

}



void TestModule::TestPlainHawkesNuclear()
{

	unsigned dim = 5, num_params = dim * (dim + 1);

	Eigen::MatrixXd B1 = (Eigen::MatrixXd::Random(dim,2).array() + 1) / 2;
	Eigen::MatrixXd B2 = (Eigen::MatrixXd::Random(dim,2).array() + 1) / 2;

	Eigen::MatrixXd B = B1 * B2.transpose() / 8;

	// std::cout << B << std::endl << std::endl;

	Eigen::EigenSolver<Eigen::MatrixXd> es(B);

	// std::cout << es.eigenvalues() << std::endl;

	OgataThinning ot(dim);

	Eigen::VectorXd params(num_params);
	
	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);

	Lambda0 = Eigen::VectorXd::Constant(dim, 0.1);
	A = B;	

	// std::cout << params << std::endl;

	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1.0);
	
	// std::cout << beta << std::endl;

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	// Store the simulated sequences
	std::vector<Sequence> sequences;

	// Simulate 10 events for each sequence
	unsigned n = 1000;
	// Simulate 2 sequences 
	unsigned num_sequences = 10;
	ot.Simulate(hawkes, n, num_sequences, sequences);

	// Print simulated sequences
	
	for(unsigned c = 0; c < sequences.size(); ++ c)
	{
		std::map<unsigned, unsigned> dim2count;
		const std::vector<Event>& seq = sequences[c].GetEvents();
		for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
		{
			// std::cout << i_event -> time << " " << i_event -> DimentionID << "; ";
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

	PlainHawkes hawkes_new(num_params, dim, beta);
	
	PlainHawkes::OPTION options;
	options.base_intensity_regularizer = NONE;
	options.excitation_regularizer = NUCLEAR;

	hawkes_new.debugfit(sequences, options, params);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	std::cout << "True Parameters : " << std::endl;
	std::cout << params.transpose() << std::endl;

}

void TestModule::TestLowRankHawkes()
{
	unsigned num_users = 64, num_items = 64;
	std::vector<Sequence> data;
	std::cout << "1. Loading " << num_users << " users " << num_items << " items " << " with 1000 events each" << std::endl;
	ImportFromExistingUserItemSequences("/Users/nandu/Development/Recommendation/sampled_entries_events", num_users, num_items, data);
	unsigned dim = num_users * num_items, num_params = 2 * dim;
	Eigen::VectorXd beta = Eigen::VectorXd::Constant(dim, 1.0);
	LowRankHawkesProcess low_rank_hawkes(num_users, num_items, beta);
	LowRankHawkesProcess::OPTION options;
	options.coefficients[LowRankHawkesProcess::LAMBDA0] = 0;
	options.coefficients[LowRankHawkesProcess::LAMBDA] = 0;

	Eigen::MatrixXd TrueLambda0, TrueAlpha;
	LoadEigenMatrixFromTxt("/Users/nandu/Development/Recommendation/truth-syn-Lambda0", num_users, num_items, TrueLambda0);
	LoadEigenMatrixFromTxt("/Users/nandu/Development/Recommendation/truth-syn-Alpha", num_users, num_items, TrueAlpha);
	Eigen::MatrixXd temp;
	LoadEigenMatrixFromTxt("/Users/nandu/Development/Recommendation/truth-syn-X0", 2 * num_users * num_items, 1, temp);
	Eigen::VectorXd X0 = temp;
	std::cout << "2. Fitting Parameters " << std::endl;
	low_rank_hawkes.debugfit(data, options, TrueLambda0, TrueAlpha, X0);
	
}

void TestModule::TestInfluenceEstimation()
{
	unsigned N = 1024;
	// Graph G("/Users/nandu/Development/exp_fit_graph/example_cascade_exp_10_network", 6, false);
	Graph G("/Users/nandu/Development/ConTinEst/std_weibull_DAG_core-1024-1-network.txt", N, true);
	Graph G1("/Users/nandu/Development/ConTinEst/std_weibull_DAG_core-1024-1-network.txt", N, false);

	G1.PrintWeibullFormatNetwork();

	unsigned num_samples = 10000, num_labels = 5;

	ConTinEst continest(&G, &G1, num_samples, num_labels);

	std::cout <<"Get all least-label lists : " << num_samples << " sets of transmission times; " << num_labels << " sets of random labels; ";
	continest.GetLeastElementLists();
	std::cout <<"done" << std::endl << std::endl;

	std::pair<unsigned, unsigned> result = G1.MaximumOutDegree();

	unsigned nodeID = result.first;
	unsigned degree = result.second;

	std::cout << "node " << nodeID << " has the largest out-degree " << degree << std::endl;

	std::set<unsigned> sources;
	sources.insert(nodeID);

	unsigned C = 10000;

	for(unsigned T = 1; T <= 10; ++ T)
	{
		std::cout << "Estimate Influence T = " << T;
		double estimated_influence = continest.EstimateNeighborhood(sources, T);
		std::cout << " done" << std::endl << std::endl;

		std::cout << "Simulation Check T = " << T << " " << " with " << C << " samples" << std::endl;

		std::cout << "ConTinEst : " << estimated_influence << std::endl << "Simulation : " << continest.RandomSimulation(T, sources, C) << std::endl << std::endl;
	}

	std::vector<double> set_T;
	std::vector<unsigned> set_K;

	set_T.push_back(10);
	set_K.push_back(10);

	std::cout <<"Influence Maximization by selecting 10 nodes with T = 10 ";

	std::vector<std::set<unsigned> > tables = continest.Optimize(set_T, set_K);

	std::cout << "done" << std::endl;

	std::cout << "selected sources : " ;

	for(std::vector<std::set<unsigned> >::const_iterator m = tables.begin(); m != tables.end(); ++ m)
	{
		for(std::set<unsigned>::const_iterator u = m->begin(); u != m->end(); ++ u)
		{
			std::cout << *u << ";";
		}
		std::cout << std::endl;
	}
}

void TestModule::TestPlot()
{
	unsigned dim = 1;
	unsigned num_params = dim * (dim + 1);

	Eigen::VectorXd params1(num_params);
	params1 << 0.1, 0.5;

	Eigen::MatrixXd beta1(dim,dim);
	beta1 << 1;

	std::vector<Sequence> sequences;

	PlainHawkes hawkes1(num_params, dim, beta1);
	hawkes1.SetParameters(params1);

	OgataThinning ot1(dim);
	sequences.clear();
	ot1.Simulate(hawkes1, 20, 1, sequences);

	hawkes1.PlotIntensityFunction(sequences[0], 0);

}
