#ifndef CONTINEST_H
#define CONTINEST_H

#include <vector>
#include <set>
#include <cstdlib>
#include <map>
#include "Graph.h"


class ConTinEst{

private:

	struct Node2Time{
		unsigned nodeID;
		double time;
		
		bool operator < (const Node2Time& n2t) const{
			if (time < n2t.time) {
				return true;
			}else {
				return false;
			}
		}
	};

	SimpleRNG RNG_;

	unsigned num_samples_;

	unsigned num_rankings_;

	Graph *G_inverse_;

	Graph *G_;

	std::vector<std::pair<float, unsigned> >*** TableList_;

	float *** keys_;


private:

	void LeastElementListsSet(float *d, const std::pair<float, unsigned> *key_node_pairs, std::vector<std::pair<float, unsigned> > *lists);

	std::set<unsigned> LZGreedy(double T, unsigned K);

	void GenerateCascade(std::vector<Node2Time>& cascade, std::set<unsigned>& initialSet, double TimeHorizon, std::map<unsigned, unsigned>& infectedBy);

public:

	ConTinEst(Graph *G_inverse, Graph *G, unsigned num_samples, unsigned num_rankings);
	~ConTinEst();

	void GetLeastElementLists();

	float EstimateNeighborhood(const std::set<unsigned>& sources, float T);

	std::vector<std::set<unsigned> > Optimize(const std::vector<double>& setT, const std::vector<unsigned>& setK);

	double RandomSimulation(double T, std::set<unsigned>& initialSet, unsigned C);

};


#endif