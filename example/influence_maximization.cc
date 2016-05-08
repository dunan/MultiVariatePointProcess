#include <iostream>
#include <Eigen/Dense>
#include "ContinEst.h"

int main(const int argc, const char** argv)
{
	unsigned N = 1024;
	Graph G("data/std_weibull_DAG_core-1024-1-network", N, true);
	Graph G1("data/std_weibull_DAG_core-1024-1-network", N, false);

	unsigned num_samples = 5000, num_labels = 5;

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

	return 0;
}