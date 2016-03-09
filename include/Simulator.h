#ifndef SIMULATOR_H
#define SIMULATOR_H
#include <vector>
#include "Process.h"
#include "Sequence.h"

/*
	
	This class defines the general Simulator Interface.
	 
*/

class Simulator
{

public:

//  This virtual function requires process-specific implementation. It simulates collection of sequences before the observation window in vec_T;
//  Parameter process stores the parameters of the specific process we are going to simulate from;
//	Parameter vec_T stores the collection of obsrvation window before which we can simualte the events. The numbef of elements in vec_T determins how many sequences we want to simulate. Each element of vec_T is the observation window wrt the respetive sequence;
//	Parameter sequences stores the simulated sequences. The number of elements in sequences is the same as that in vec_T; 
	virtual void Simulate(IProcess& process, const std::vector<double>& vec_T, std::vector<Sequence>& sequences) = 0;

//  This virtual function requires process-specific implementation. It simulates collection of sequences.
//	Parameter process stores the parameters of the specific process we are going to simulate from;
//	Parameter n is the number of events we are going to simualte in each sequence;
//	Parameter num_sequences is the number of sequence we are going to simulate;
//	Parameter sequences stores the simulated sequences. The number of elements in sequences is the same as num_sequences;
	virtual void Simulate(IProcess& process, const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences) = 0;

//  This virtual function requires process-specific implementation. It returns the next simulated event
//	Parameter process stores the parameters of the specific process we are going to simulate from;
//	Parameter data is the given history. 
	virtual Event SimulateNext(IProcess& process, const Sequence& data) = 0;



};
#endif