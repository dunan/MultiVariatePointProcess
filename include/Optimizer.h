#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include "Process.h"
#include "SimpleRNG.h"

class Optimizer
{
private:

	//  Internal implementation for random number generator;
	SimpleRNG RNG_;

public:

	Optimizer()
	{
		RNG_.SetState(0, 0);
	}

	void SGD(IProcess* process, const double& gamma0, const unsigned& ini_max_iter, const std::vector<Sequence>& data);

};

#endif