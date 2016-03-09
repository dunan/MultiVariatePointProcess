#ifndef OGATA_THINNING_H
#define OGATA_THINNING_H
#include <vector>
#include "../include/Simulator.h"
#include "../include/Process.h"
#include "../include/SimpleRNG.h"

/*
	
	This class defines the general Simulator based on Ogata's Thinning algorithm.
	 
*/

class OgataThinning : Simulator
{

private:

//	Records the number of dimensions for internal use;
	unsigned num_dims_;

// 	Internal implementation for random number generator;
	SimpleRNG RNG_;

public:

//  Constructor : num_dims is the number of dimensions we are going to simulate;
	OgataThinning(const unsigned& num_dims) : Simulator(), num_dims_(num_dims)
	{
		// Initialze the random generator;
		RNG_.SetState(0, 0);
	}


//  This virtual function requires process-specific implementation. It simulates collection of sequences before the observation window in vec_T;
//  Parameter process stores the parameters of the specific process we are going to simulate from;
//	Parameter vec_T stores the collection of obsrvation window before which we can simualte the events. The numbef of elements in vec_T determins how many sequences we want to simulate. Each element of vec_T is the observation window wrt the respetive sequence;
//	Parameter sequences stores the simulated sequences. The number of elements in sequences is the same as that in vec_T; 
	virtual void Simulate(IProcess& process, const std::vector<double>& vec_T, std::vector<Sequence>& sequences);

//  This virtual function requires process-specific implementation. It simulates collection of sequences.
//	Parameter process stores the parameters of the specific process we are going to simulate from;
//	Parameter n is the number of events we are going to simualte in each sequence;
//	Parameter num_sequences is the number of sequence we are going to simulate;
//	Parameter sequences stores the simulated sequences. The number of elements in sequences is the same as num_sequences;
	virtual void Simulate(IProcess& process, const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences);

//  This virtual function requires process-specific implementation. It returns the next simulated event
//	Parameter process stores the parameters of the specific process we are going to simulate from;
//	Parameter data is the given history. 
	virtual Event SimulateNext(IProcess& process, const Sequence& data);

};
#endif