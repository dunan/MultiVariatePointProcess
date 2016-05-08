/**
 * \file Graph.cc
 * \brief The class implementation of Graph.
 */
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstdlib>
#include <iomanip>

#include "../include/Graph.h"
#include "../include/Utility.h"


void Graph::PrintWeibullFormatNetwork()
{
	if(edge_parameter.size() > 0)
	{
		for (unsigned i = 0; i < nodes.size(); ++ i) 
		{
			for (std::set<unsigned>::iterator u = nodes[i].children.begin(); u != nodes[i].children.end(); ++ u) 
			{
				std::cout << i << "\t" << *u << "\t" << std::fixed << std::setprecision(4) << std::setw(5) << edge_parameter[i][*u].scale << "\t" <<std::setprecision(4) << std::setw(5) << edge_parameter[i][*u].shape<< std::endl;
			}
		}
	}else
	{
		std::cout << "No edge parameters. Only print the network structure." << std::endl;

		for (unsigned i = 0; i < nodes.size(); ++ i) 
		{
			for (std::set<unsigned>::iterator u = nodes[i].children.begin(); u != nodes[i].children.end(); ++ u) 
			{
				std::cout << i << "\t" << *u << std::endl;
			}
		}
	}
}

void Graph::LoadWeibullFormatNetwork(const std::string& filename, std::string splitter, bool reverse)
{

	nodes.reserve(N);

	std::stringstream ss;
	for (unsigned i = 0; i < N; ++ i) {
		
		Node node;
		nodes.push_back(node);
	
	}
	
	std::ifstream fin(filename.c_str(),std::ios::in);
	std::string str = "",str1 = "",str2 = "", str3 = "", str4 = "";
	unsigned readline = 1, count = 0;
	while(getline(fin,str))
	{
		
		if(readline >= N + 2)
		{
			std::vector<std::string> line = SeperateLineWordsVector(str,splitter);
			str1 = line[0];
			str2 = line[1];

			if(line.size() == 4)
			{
				str3 = line[2];
				str4 = line[3];	
			}
			
			// if(str1 == str2)
			// {
			// 	std::cout <<str<<std::endl;
			// }
			// else 
			{
				
				int idx1, idx2;
				
				if(!reverse)
				{
					idx1 = std::atoi(str1.c_str());
					idx2 = std::atoi(str2.c_str());
				}else {
					idx2 = std::atoi(str1.c_str());
					idx1 = std::atoi(str2.c_str());
				}

				nodes[idx1].children.insert(idx2);
				nodes[idx2].parents.insert(idx1);
				
				if(line.size() == 4)
				{
					Parameter param;
					param.shape = std::atof(str3.c_str());
					param.scale = std::atof(str4.c_str());
					
					if(edge_parameter.find(idx1) == edge_parameter.end())
					{
						std::map<unsigned, Parameter> temp;
						temp.insert(std::make_pair(idx2, param));
						
						edge_parameter.insert(std::make_pair(idx1, temp));
						
					}else {
						edge_parameter[idx1].insert(std::make_pair(idx2, param));
					}
				}
			}
			
			++ count;
			
		}
		
		++ readline;
		
	}
	
	fin.close();
}


void Graph::SampleEdgeWeightWbl()
{
	for (unsigned i = 0; i < nodes.size(); ++ i) 
	{
		for (std::set<unsigned>::iterator c = nodes[i].children.begin(); c != nodes[i].children.end(); ++ c) {
			
			const Parameter &param = edge_parameter[i][*c];

			edge_weight[i][*c] = RNG_.GetWeibull(param.shape, param.scale);
			
		}

	}
}

std::pair<unsigned, unsigned> Graph::MaximumOutDegree()
{
	unsigned d = 0;
	unsigned mark = 0;
	
	for (unsigned i = 0; i < nodes.size(); ++ i) {
		unsigned tmp = nodes[i].children.size();
		if(d < tmp)
		{
			d = tmp;
			mark = i;
		}
		
	}
	return std::make_pair(mark, d);	
}

