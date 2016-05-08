/**
 * \file Event.h
 * \brief The class definition of Event.
 */
#ifndef EVENT_H
#define EVENT_H

/**
 * \class Event Event.h "include/Event.h" Defines a simple structure for describing an event point.
 * \brief Event contains simple attributes of an event point.
 */
class Event
{

public:

/**
 * 	\brief The unique ID in the current sequence.
 */
	int EventID;

/**
 * 	\brief The unique ID of the sequence this event is associated with.
 */
	int SequenceID;

/**
 * 	\brief The unique ID of the dimension this event is associated with.
 */
	int DimentionID;

/**
 * 	\brief The time when this event occurs.
 */
	double time;

/**
 * 	\brief The marker ID associated with this event.
 */
	int marker;

};

#endif