/**
 * \file Diagnosis.h
 * \brief The class definition of Diagnosis for the basic residual analysis.
 */
#ifndef DIAGNOSIS_H
#define DIAGNOSIS_H

#include "Process.h"
#include "Sequence.h"

/**
 * \class Diagnosis Diagnosis.h "include/Diagnosis.h"
 * \brief Diagnosis implements the basic residual analysis for a given general point process.
 */
class Diagnosis
{

public:

/**
 * \brief Basic residual analysis fitting.
 * 
 * @param  process a given point process.
 * @param  data    sequence of past events.
 * @return         the residual analysis fitting. If the given point process can explain the sequence of events well, the returned value should be close to 1. 
 */
	static double TimeChangeFit(IProcess& process, const Sequence& data);

};

#endif