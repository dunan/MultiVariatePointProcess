#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <vector>
#include <cstdio>
#include "Event.h"

/*
	
	This class defines the general sequence class.
	 
*/

class Sequence
{

private:

//  Internally, we represent each sequence as a vector of Event
	std::vector<Event> sequence_;

//  Observation window of this process;
	double T_;

public:

//  Constructor : T is the observation window;
	Sequence(const double& T) : T_(T){}
	Sequence() : T_(0){}

//  Add an event to the current sequence. 
	void Add(const Event& event) 
	{
		sequence_.push_back(event);

// 	If the current maximum time is greater than the predefined observation time window, we update the time window to the maximum event time observed so far.
		T_ = event.time >= T_ ? event.time : T_;
	}

//  Return a constant reference to the sequence of event;
	const std::vector<Event>& GetEvents() const {return sequence_;}

//  Return the observation window;
	double GetTimeWindow() const {return T_;}

};

#endif