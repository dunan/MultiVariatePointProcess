#ifndef DIAGNOSIS_H
#define DIAGNOSIS_H

#include "Process.h"
#include "Sequence.h"

class Diagnosis
{

public:

	static double TimeChangeFit(IProcess& process, const Sequence& data);

};

#endif