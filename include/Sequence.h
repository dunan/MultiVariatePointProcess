/**
 * \file Sequence.h
 * \brief The class definition of Sequence.
 */
#ifndef SEQUENCE_H
#define SEQUENCE_H
#include <iostream>
#include <vector>
#include <cstdio>
#include "Event.h"

/**
 * \class Sequence Sequence.h "include/Sequence.h"
 * \brief Sequence encapsulates the operations on a sequence of events.
 */
class Sequence
{

private:

/**
 * \brief Internal representation of a sequence of events.
 */
	std::vector<Event> sequence_;

/**
 * \brief Observation window of the sequence.
 */
	double T_;

public:

/**
 * The constructor.
 *
 * @param[in] T observation window of the sequence.
 */
	Sequence(const double& T) : T_(T){}
	Sequence() : T_(0){}

/**
 * \brief Add a new event to the current sequence.
 * @param event a given new event.
 */
	void Add(const Event& event) 
	{
		sequence_.push_back(event);

/**
 * If the current maximum time is greater than the predefined observation time window, we update the time window to the maximum event time observed so far.
 */
		T_ = event.time >= T_ ? event.time : T_;
	}

/**
 * \brief Return a constant reference to the sequence of event.
 */
	const std::vector<Event>& GetEvents() const {return sequence_;}
 
/**
 * \brief Return the observation window.
 * @return the observation window of the sequence.
 */
	double GetTimeWindow() const {return T_;}

/**
 * \brief Update the current observation window.
 * @param T new observation window.
 */
	void SetTimeWindow(const double& T) {T_ = T;}

/**
 * \brief Get rid of the last event in the current sequence.
 */
	void PopBack(){
		sequence_.pop_back();
	};

};

#endif