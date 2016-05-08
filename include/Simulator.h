/**
 * \file Simulator.h
 * \brief The class definition of Simulator which defines the general interface of a point process simulator.
 */
#ifndef SIMULATOR_H
#define SIMULATOR_H
#include <vector>
#include "Process.h"
#include "Sequence.h"

/**
 * \class Simulator Simulator.h "include/Simulator.h"
 * \brief Simulator defines a general simulator for point processes.
 */
class Simulator
{

public:

/**
 * Simulates collection of sequences before the observation window in vec_T;
 * @param[in] process   the parameters of the specific process we are going to simulate from.
 * @param[in] vec_T     the collection of obsrvation window before which we can simualte the events. The numbef of elements in vec_T determins how many sequences we want to simulate. Each element of vec_T is the observation window wrt the respetive sequence.
 * @param[out] sequences the simulated sequences. The number of elements in sequences is the same as that in vec_T; 
 */
	virtual void Simulate(IProcess& process, const std::vector<double>& vec_T, std::vector<Sequence>& sequences) = 0;

/**
 * Simulates collection of sequences, each of which has \f$n\f$ events.
 * @param process       the parameters of the specific process we are going to simulate from.
 * @param n             the number of events we are going to simualte in each sequence.
 * @param num_sequences the number of sequence we are going to simulate.
 * @param sequences     the simulated sequences. The number of elements in sequences is the same as num_sequences.
 */
	virtual void Simulate(IProcess& process, const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences) = 0;

/**
 * Simulates the next single event.
 * @param  process the parameters of the specific process we are going to simulate from.
 * @param  data    the sequence of past events.
 * @return         the simulated next event.
 */
	virtual Event SimulateNext(IProcess& process, const Sequence& data) = 0;

};
#endif