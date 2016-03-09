#ifndef EVENT_H
#define EVENT_H


/*
	
	This class defines the general Event class.
	 
*/

class Event
{

public:

//  The unique ID in the current sequence;
	int EventID;

//  The unique ID of the sequence this event is associated with;
	int SequenceID;

//  The unique ID of the dimension this event is associated with;
	int DimentionID;

//  The event time;
	double time;

//  The event marker;
	int marker;

};

#endif