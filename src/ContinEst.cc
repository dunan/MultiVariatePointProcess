/**
 * \file ConTinEst.cc
 * \brief The class implementation of ConTinEst for scalable continuous-time influence estimation.
 */
#include <map>
#include <algorithm>
#include "ContinEst.h"

ConTinEst::ConTinEst(Graph *G_inverse, Graph *G, unsigned num_samples, unsigned num_rankings)
{
	RNG_.SetState(0, 0);

	num_samples_ = num_samples;

	num_rankings_ = num_rankings;

	G_inverse_ = G_inverse;

	G_ = G;

	unsigned N = G_inverse_->N;

	TableList_ = new std::vector<std::pair<float, unsigned> >**[num_samples_];
	keys_ = new float**[num_samples_];

	for (unsigned i = 0; i < num_samples_; ++i)
	{
		TableList_[i] = new std::vector<std::pair<float, unsigned> >*[num_rankings_];
		keys_[i] = new float*[num_rankings_];

		for (unsigned j = 0; j < num_rankings_; ++j)
		{
			TableList_[i][j] = new std::vector<std::pair<float, unsigned> >[N];
			keys_[i][j] = new float[N];
		}
	}
}

ConTinEst::~ConTinEst()
{
	unsigned N = G_inverse_->N;
	
	for (unsigned i = 0; i < num_samples_; ++i)
	{
		for (unsigned j = 0; j < num_rankings_; ++j)
		{
			for(unsigned k = 0; k < N; ++ k)
			{
				std::vector<std::pair<float, unsigned> >().swap(TableList_[i][j][k]);
			}
			delete[] TableList_[i][j];
			delete[] keys_[i][j];
		}
		delete[] TableList_[i];
		delete[] keys_[i];
	}

	delete[] TableList_;
	delete[] keys_;

	TableList_ = NULL;
	keys_ = NULL;
}

void ConTinEst::LeastElementListsSet(float *d, const std::pair<float, unsigned> *key_node_pairs, std::vector<std::pair<float, unsigned> > *lists)
{
	unsigned N = G_inverse_->N;

	std::pair<float, unsigned> bound(-1,N);

	for(unsigned i = 0; i < N; ++ i)
	{
		unsigned vi = key_node_pairs[i].second;

		std::set<std::pair<float, unsigned> > key_to_node;
		std::map<unsigned, float> node_to_key;

		key_to_node.insert(std::make_pair(0, vi));
		node_to_key.insert(std::make_pair(vi, 0));

		while(!key_to_node.empty())
		{

			std::set<std::pair<float, unsigned> >::iterator itlow = key_to_node.lower_bound(bound);
			std::map<unsigned, float>::iterator itlow_con = node_to_key.find(itlow->second);

			float dk = itlow->first;
			unsigned vk = itlow->second;
			key_to_node.erase(itlow);
			node_to_key.erase(itlow_con);

			lists[vk].push_back(std::make_pair(dk, vi));
			d[vk] = dk;

			for (std::set<unsigned>::iterator c = G_inverse_->nodes[vk].children.begin(); c != G_inverse_->nodes[vk].children.end(); ++ c) {

				unsigned vj = *c;


				itlow_con = node_to_key.find(*c);

				float tmp = dk + G_inverse_->edge_weight[vk][vj];
				if(itlow_con != node_to_key.end())
				{
					if(tmp < itlow_con->second)
					{
						itlow = key_to_node.find(std::make_pair(itlow_con->second, itlow_con->first));

						key_to_node.erase(itlow);
						key_to_node.insert(std::make_pair(tmp,itlow_con->first));
						itlow_con->second = tmp;

					}
				}else if(tmp < d[vj])
				{

					key_to_node.insert(std::make_pair(tmp, vj));
					node_to_key.insert(std::make_pair(vj, tmp));

				}


			}
			

		}


	}
}

void ConTinEst::GetLeastElementLists()
{
	unsigned N = G_inverse_->N;

	std::pair<float, unsigned> *key_node_pairs = new std::pair<float, unsigned>[N];

	float *d = new float[N]; 
	
	for (unsigned i = 0; i < num_samples_; ++ i) {

		G_inverse_->SampleEdgeWeightWbl();

		for (unsigned j = 0; j < num_rankings_; ++ j) {
		
			// initialize keys and distance d
			for (unsigned k = 0; k < N; ++ k) {
				float key = RNG_.GetExponential(1.0);
				keys_[i][j][k] = key;
				key_node_pairs[k].first = key;
				key_node_pairs[k].second = k;
			}

			std::sort(key_node_pairs, key_node_pairs + N);
			std::fill_n(d, N, 1e10);

			LeastElementListsSet(d, key_node_pairs, TableList_[i][j]);
		}

	}

	delete[] d;
	delete[] key_node_pairs;	
}

float ConTinEst::EstimateNeighborhood(const std::set<unsigned>& sources, float T)
{

	float size = 0.0, avg = 0.0;
	unsigned N = G_inverse_->N;

	if(!sources.empty())
	{
		for (unsigned i = 0; i < num_samples_; ++ i) {

			size = 0.0;

			for (unsigned j = 0; j < num_rankings_; ++ j) {

				float minRank = 1e10;
				std::pair<float, unsigned> tmp(T, N);
				
				for (std::set<unsigned>::const_iterator s = sources.begin(); s != sources.end(); ++ s) {
				
					std::vector<std::pair<float, unsigned> >::iterator idx = lower_bound(TableList_[i][j][*s].begin(), TableList_[i][j][*s].end(), tmp, std::greater<std::pair<float, unsigned> >());
				
					if (keys_[i][j][idx->second] < minRank) {
						minRank = keys_[i][j][idx->second];
					}
				
				}

				size += minRank;

			}

			avg += ((num_rankings_ - 1) / size);

		}

		return avg / num_samples_;
	}else
	{
		return 0;
	}

}


std::set<unsigned> ConTinEst::LZGreedy(double T, unsigned K)
{
	unsigned N = G_inverse_->N;

	std::set<unsigned> sources;

	std::vector<std::pair<double, unsigned> > marginal_gain;
	make_heap(marginal_gain.begin(), marginal_gain.end());
	
	for (unsigned i = 0; i < N; ++i)
	{
		std::set<unsigned> tmp;
		tmp.insert(i);

		marginal_gain.push_back(std::make_pair(EstimateNeighborhood(tmp, T), i));
		push_heap(marginal_gain.begin(), marginal_gain.end());

	}

	std::pair<double, unsigned> &top_max = marginal_gain.front();

	double total = top_max.first;
	sources.insert(top_max.second);

	pop_heap(marginal_gain.begin(), marginal_gain.end());
	marginal_gain.pop_back();

	bool *valid = new bool[N];

	while(sources.size() < K)
	{
		std::fill_n(valid, N, false);

		while(true)
		{
			top_max = marginal_gain.front();

			if (valid[top_max.second])	
			{
				sources.insert(top_max.second);
				total += top_max.first;
				pop_heap(marginal_gain.begin(), marginal_gain.end());
				marginal_gain.pop_back();
				break;
			}

			std::set<unsigned> tmp = sources;
			tmp.insert(top_max.second);

			double gain = EstimateNeighborhood(tmp, T) - total;

			top_max.first = (gain > 0) ? gain : 0;
			valid[top_max.second] = true;
			make_heap(marginal_gain.begin(), marginal_gain.end());

		}
	}

	// for(std::set<unsigned>::const_iterator s = sources.begin(); s != sources.end(); ++ s)
	// {
	// 	cout << *s << " ";
	// }
	// cout << endl;


	delete[] valid;

	return sources;
}

std::vector<std::set<unsigned> > ConTinEst::Optimize(const std::vector<double>& setT, const std::vector<unsigned>& setK)
{
	std::vector<std::set<unsigned> > sources;
	for(std::vector<unsigned>::const_iterator k = setK.begin(); k != setK.end(); ++ k)
	{
		for(std::vector<double>::const_iterator t = setT.begin(); t != setT.end(); ++ t)
		{
			sources.push_back(LZGreedy(*t, *k));
		}
	}
	
	return sources;	
}

void ConTinEst::GenerateCascade(std::set<unsigned>& initialSet, double TimeHorizon, std::map<unsigned, unsigned>& infectedBy, std::vector<Node2Time>& cascade)
{
	double GlobalTime = 0.0;
	
	std::vector<Node2Time> visitedNodes;
	
	for (std::set<unsigned>::iterator u = initialSet.begin(); u != initialSet.end(); ++ u) {
		Node2Time n2t;
		n2t.nodeID = *u;
		n2t.time = GlobalTime;
		visitedNodes.push_back(n2t);
	}
	
	while (true) {
		
		std::sort(visitedNodes.begin(), visitedNodes.end());
		
		
		std::map<unsigned, unsigned> visitedNodes2Idx;
		for (unsigned i = 0; i < visitedNodes.size(); ++ i) {
			visitedNodes2Idx.insert(std::make_pair(visitedNodes[i].nodeID, i));
		}
		
		unsigned currentID = visitedNodes[0].nodeID;
		GlobalTime = visitedNodes[0].time;
		if(GlobalTime >= TimeHorizon)
		{
			break;
		}
		
		cascade.push_back(visitedNodes[0]);
		
		for (std::set<unsigned>::iterator u = G_->nodes[currentID].children.begin(); u != G_->nodes[currentID].children.end(); ++ u) {
			
			
			if((infectedBy.find(currentID) != infectedBy.end()) && (infectedBy[currentID] == *u))
			{
				continue;
			}
			
			double deltaT = RNG_.GetWeibull(G_->edge_parameter[currentID][*u].shape,G_->edge_parameter[currentID][*u].scale);
					
			double t1 = GlobalTime + deltaT;
			
			std::map<unsigned, unsigned>::iterator m = visitedNodes2Idx.find(*u);
			
			if(m != visitedNodes2Idx.end())
			{
				double t2 = visitedNodes[m->second].time;
				if((t2 != TimeHorizon) && (t2 > t1))
				{
					visitedNodes[m->second].time = t1;
					infectedBy[*u] = currentID;
				}
			}else {
				Node2Time n2t;
				n2t.nodeID = *u;
				n2t.time = t1;
				
				visitedNodes2Idx.insert(std::make_pair(*u, visitedNodes.size()));
				visitedNodes.push_back(n2t);
				infectedBy.insert(std::make_pair(*u, currentID));
			}
			
		}
		
		visitedNodes[0].time = TimeHorizon;
		
	}
}

double ConTinEst::RandomSimulation(double T, std::set<unsigned>& initialSet, unsigned C)
{
	std::vector<Node2Time> cascade;
	
	std::vector<double> Node2Infected(G_->N, 0);
	
	std::map<unsigned, unsigned> infectedBy;
	
	for (unsigned c = 0; c < C; ++ c) {
		
		cascade.clear();
		
		infectedBy.clear();
		
		if (initialSet.size() > 0) {
			
			GenerateCascade(initialSet, T, infectedBy, cascade);
			
			for (int i = 0; i < cascade.size(); ++ i) {
				if (cascade[i].time <= T) {
					Node2Infected[cascade[i].nodeID] += 1;
				}
			}
		}
		
		
	}
	
	for (unsigned i = 0; i < Node2Infected.size(); ++ i) {
		Node2Infected[i] = Node2Infected[i] / C;
	}

	double sum = 0;
	for (std::vector<double>::const_iterator j = Node2Infected.begin(); j != Node2Infected.end(); ++j)
	{
		sum += *j;
	}

	return sum;
}
