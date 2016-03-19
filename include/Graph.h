/*

Definition of the Graph Class.  

Author : Nan Du (dunan@gatech.edu)

*/

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
	
	Graph(std::string g_filename, unsigned numNodes);
	~Graph(){};

	std::vector<Node> nodes;
	std::map<unsigned, std::map<unsigned, float> > edge_weight;
	std::map<unsigned, std::map<unsigned, Parameter> > edge_parameter;

	void PrintWblNetwork();

	void LoadWeibullFormatNetwork(std::string splitter, bool reverse);

	void SampleEdgeWeightWbl();

	std::pair<unsigned, unsigned> MaximumOutDegree();

private:

	std::string filename;

	SimpleRNG RNG;

};

#endif