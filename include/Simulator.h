#ifndef SIMULATOR_H
#define SIMULATOR_H
#include <vector>
#include "Process.h"
#include "Sequence.h"

class Simulator
{

public:

	virtual void Simulate(IProcess& process, const std::vector<double>& vec_T, std::vector<Sequence>& sequences) = 0;

	virtual void Simulate(IProcess& process, const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences) = 0;

	// virtual void Simulate(IProcess& process, const unsigned& n, const unsigned& D) = 0;

};
#endif