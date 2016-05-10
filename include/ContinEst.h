/**
 * \file ConTinEst.h
 * \brief The class definition of ConTinEst for scalable continuous-time influence estimation.
 */
#ifndef CONTINEST_H
#define CONTINEST_H

#include <vector>
#include <set>
#include <cstdlib>
#include <map>
#include "Graph.h"

/**
 * \class ConTinEst ConTinEst.h "include/ConTinEst.h"
 * \brief ConTinEst implements the scalable influence estimation algorithm. 
 * 
 * Check out the following paper for more details.
 * - [Scalable Influence Estimation in Continuous-Time Diffusion Networks](http://www.cc.gatech.edu/~ndu8/pdf/DuSonZhaMan-NIPS-2013.pdf). Nan Du, Le Song, Manuel Gomez Rodriguez, and Hongyuan Zha. Neural Information Processing Systems (NIPS). Dec. 5 - Dec. 10, 2013, Lake Tahoe, Nevada, USA.
 *
 * The current version does not include the MPI implementaion, so the memory consumption will be large when working with networks of millions of nodes. 
 */
class ConTinEst{

private:

/**
 * \brief A nodeID and infection time pair which can be sorted by the ascending order of time.
 */
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

/**
 * \brief Internal implementation of simple random number generator.
 */
	SimpleRNG RNG_;

/**
 * \brief Number of sampled edge-weight sets.
 */
	unsigned num_samples_;

/**
 * \brief Number of sampled node-label sets.
 */
	unsigned num_rankings_;

/**
 * \brief Graph object with each edge being reversed.
 */
	Graph *G_inverse_;
/**
 * \brief Graph object representing the diffusion network structure.
 */
	Graph *G_;

/**
 * \brief Least-label list data structure.
 *
 * TableList_[n][m][i] stores the least-label list for node \f$i\f$ in the \f$n\f$-th sampled weighted network with the \f$m\f$-th sampled set of node-labels. Each element in the least-label list is a pair of (node-label, node-index).
 */
	std::vector<std::pair<float, unsigned> >*** TableList_;

/**
 * \brief Collection of node labels.
 *
 * keys_[n][m][i] is the node label of node \f$i\f$ in the \f$n\f$-th sampled weighted network with the \f$m\f$-th sampled set of node-labels.
 */
	float *** keys_;

private:

/**
 * \brief Calculate the least-label list for each node.
 * @param[in] d              distance of each node to the source.
 * @param[in] key_node_pairs node label and node index pairs.
 * @param[out] lists          least-label list for each node.
 */
	void LeastElementListsSet(float *d, const std::pair<float, unsigned> *key_node_pairs, std::vector<std::pair<float, unsigned> > *lists);

/**
 * \brief The greedy algorithm with lazy evaluation.
 * @param[in] T given observation window.
 * @param[i] K maximum number of selected sources.
 * @return	the set of selected sources that maximize the influence by the given time \f$T\f$.
 */
	std::set<unsigned> LZGreedy(double T, unsigned K);

/**
 * \brief Simulate sequences of infection time according to the continuous-time independent cascade model.
 * @param initialSet  set of source nodes. 
 * @param TimeHorizon observation time window.
 * @param infectedBy  records who has infected whom in the simulated sequence.
 * @param cascade     a simulated sequence.
 */
	void GenerateCascade(std::set<unsigned>& initialSet, double TimeHorizon, std::map<unsigned, unsigned>& infectedBy, std::vector<Node2Time>& cascade);

public:

/**
 * \brief The constructor.
 *
 * @param[i] G_inverse Graph object storing the given diffusion network with each edge being reversed.
 * @param[i] G Graph object storing the given diffusion network.
 * @param[i] num_samples the number of sampled weighted networks.
 * @param[i] num_rankings the number of sampled collections of node labels on a given sampled weighted network.
 */
	ConTinEst(Graph *G_inverse, Graph *G, unsigned num_samples, unsigned num_rankings);
	~ConTinEst();

/**
 * \brief Calculate the least-label lists.
 */
	void GetLeastElementLists();

/**
 * \brief Estimate the neighboorhood size of a given set of sources.
 * @param  sources a set of source nodes.
 * @param  T       observation window.
 * @return         the estimated neighboorhood size of the given source set.
 */
	float EstimateNeighborhood(const std::set<unsigned>& sources, float T);

/**
 * \brief Continuous-time influence maximization.
 * @param[in] setT a set of observation windows.
 * @param[in] setK a set of maximum source-set sizes.
 * @return	the collection of sets of selected sources.
 */
	std::vector<std::set<unsigned> > Optimize(const std::vector<double>& setT, const std::vector<unsigned>& setK);

/**
 * \brief Continuous-time influence maximization by Monte-carlo simulations.
 * @param  T          observation window.
 * @param  initialSet a set of source nodes.
 * @param  C          the maximum number of selected sources.	
 * @return            the estimated influence of the given source set.
 */
	double RandomSimulation(double T, std::set<unsigned>& initialSet, unsigned C);

};


#endif