#ifndef GRAPH_H
#define GRAPH_H
#include <vector>
#include <map>
#include <string>
#include <set>
#include "SimpleRNG.h"

class Graph{
	
public:

	struct Node {

		std::set<unsigned> children;
		std::set<unsigned> parents;
	
	};

	struct Parameter { 
		float scale;
		float shape;
	};

	unsigned N;
	
	Graph(const std::string& graph_filename, unsigned num_nodes, bool reverse) : N(num_nodes)
	{
		RNG_.SetState(0, 0);
		LoadWeibullFormatNetwork(graph_filename, ",", reverse);
	}
	~Graph(){}

	std::vector<Node> nodes;
	std::map<unsigned, std::map<unsigned, float> > edge_weight;
	std::map<unsigned, std::map<unsigned, Parameter> > edge_parameter;

	void PrintWeibullFormatNetwork();

	void SampleEdgeWeightWbl();

	std::pair<unsigned, unsigned> MaximumOutDegree();

private:

	SimpleRNG RNG_;

	void LoadWeibullFormatNetwork(const std::string& filename, std::string splitter, bool reverse);

};

#endif