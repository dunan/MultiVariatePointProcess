/**
 * \file Graph.h
 * \brief The class definition of Graph.
 */
#ifndef GRAPH_H
#define GRAPH_H
#include <vector>
#include <map>
#include <string>
#include <set>
#include "SimpleRNG.h"

/**
 * \class Graph Graph.h "include/Graph.h"
 * \brief Graph represents a given diffusion network structure.
 */
class Graph{
	
public:

/**
 * \brief Defines a single node object.
 */
	struct Node {
/**
 * \brief The set of child nodes the current node points to.
 */
		std::set<unsigned> children;
/**
 * \brief The set of parent nodes pointing to the current node.
 */
		std::set<unsigned> parents;
	};

/**
 * \brief The parameter of the pairwise Weibull distribution for the respective diffusion time.
 */
	struct Parameter { 
		/**
		 * \brief The scale parameter of the pairwise Weibull distributions.
		 */
		float scale;
		/**
		 * \brief The shape parameter of the pairwise Weibull distributions.
		 */
		float shape;
	};

/**
 * \brief The total number of nodes of a diffusion network.
 */
	unsigned N;
	
/**
 * \brief The constructor.
 *
 * @param[in] graph_filename the filename of the input diffusion network.
 * @param[in] num_nodes the total number of nodes in a diffusion network.
 * @param[in] reverse whether the edge direction should be reversed or not.
 */
	Graph(const std::string& graph_filename, unsigned num_nodes, bool reverse) : N(num_nodes)
	{
		RNG_.SetState(0, 0);
		LoadWeibullFormatNetwork(graph_filename, ",", reverse);
	}
	~Graph(){}

/**
 * \brief The set of nodes of the given diffusion network.
 */
	std::vector<Node> nodes;
/**
 * \brief edge_weight[i][j] stores a sampled diffusion time from node \f$i\f$ to node \f$j\f$.
 */
	std::map<unsigned, std::map<unsigned, float> > edge_weight;
/**
 * \brief edge_parameter[i][j] stores the parameter of the Weibull distribution along the edge from node \f$i\f$ to node \f$j\f$.
 */
	std::map<unsigned, std::map<unsigned, Parameter> > edge_parameter;

/**
 * \brief Prints the edges of the loaded diffusion network structure.
 *
 * The output format is defined as: node index \f$i\f$,node index \f$j\f$,edge_parameter[i][j].scale,edge_parameter[i][j].shape
 */
	void PrintWeibullFormatNetwork();

/**
 * \brief Samples the pairwise diffusion time from the corresponding Weibull distribution.
 */
	void SampleEdgeWeightWbl();

/**
 * \brief Gets the node with the maximum degree.
 * @return the node with the maximum degree and the degree value.
 */
	std::pair<unsigned, unsigned> MaximumOutDegree();

private:

/**
 * \brief Internal implmentation of simple random generator.
 */
	SimpleRNG RNG_;

/**
 * \brief Loads the diffusion network with pairwise Weibull distribution for the diffusion time.
 * @param filename network filename.
 * @param splitter seperator between inputs per line.
 * @param reverse  whether to reverse the edge direction or not.
 */
	void LoadWeibullFormatNetwork(const std::string& filename, std::string splitter, bool reverse);

};

#endif