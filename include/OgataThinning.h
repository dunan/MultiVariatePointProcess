#ifndef OGATA_THINNING_H
#define OGATA_THINNING_H
#include <vector>
#include "../include/Simulator.h"
#include "../include/Process.h"
#include "../include/SimpleRNG.h"

class OgataThinning : Simulator
{

private:

	unsigned num_dims_;

	SimpleRNG RNG_;

public:

	OgataThinning(const unsigned& num_dims) : Simulator(), num_dims_(num_dims)
	{
		RNG_.SetState(0, 0);
	}

	virtual void Simulate(IProcess& process, const std::vector<double>& vec_T, std::vector<Sequence>& sequences);

	virtual void Simulate(IProcess& process, const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences);

};
#endif