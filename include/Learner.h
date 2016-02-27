#include <vector>
#include "Process.h"
#include "Sequence.h"

/*
	
	This class defines the general Learner Interface.
	 
*/

class UniversalLearner
{

public:

//  Fit the given process with respect to the given data;
	virtual void fit(IProcess& process, std::vector<Sequence>& data) = 0;
	
};