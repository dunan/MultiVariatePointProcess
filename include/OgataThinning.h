/**
 * \file OgataThinning.h
 * \brief The class definition of OgataThinning implementing Ogata's thinning algorithm.
 */
#ifndef OGATA_THINNING_H
#define OGATA_THINNING_H
#include <vector>
#include "../include/Simulator.h"
#include "../include/Process.h"
#include "../include/SimpleRNG.h"

/**
 * \class OgataThinning OgataThinning.h "include/OgataThinning.h"
 * \brief This class implements the general Simulator based on Ogata's Thinning algorithm.
 */
class OgataThinning : public Simulator
{

private:

/**
 * Records the number of dimensions for internal use.
 */
	unsigned num_dims_;

/**
 * Step we simulate into the future.
 */
	double step_;	
/**
 * Internal implementation for random number generator.
 */
	SimpleRNG RNG_;

public:

/**
 * The constructor.
 *
 * @param[in] num_dims the number of dimension of the given point process.
 */
	OgataThinning(const unsigned& num_dims) : Simulator(), num_dims_(num_dims)
	{
		step_ = 1.0;

		// Initialze the random generator;
		RNG_.SetState(314, 314);
	}

	virtual void Simulate(IProcess& process, const std::vector<double>& vec_T, std::vector<Sequence>& sequences);

	virtual void Simulate(IProcess& process, const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences);

	virtual Event SimulateNext(IProcess& process, const Sequence& data);

};
#endif