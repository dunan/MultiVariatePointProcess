#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <vector>
#include <cstdio>
#include "Event.h"

class Sequence
{

private:

	std::vector<Event> sequence_;

	double T_;

public:

	Sequence(const double& T) : T_(T){}
	Sequence() : T_(0){}

	void Add(const Event& event) 
	{
		sequence_.push_back(event);

		T_ = event.time >= T_ ? event.time : T_;
	}

	const std::vector<Event>& GetEvents() const {return sequence_;}

	double GetTimeWindow() const {return T_;}

};

#endif