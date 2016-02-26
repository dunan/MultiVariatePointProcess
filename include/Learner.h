#include <vector>
#include "Process.h"
#include "Sequence.h"

class UniversalLearner
{

public:

	virtual void fit(IProcess& process, std::vector<Sequence>& data) = 0;
	
};