#include <vector>
#include <Process.h>
#include <Sequence.h>

class Simulator
{

public:

	virtual std::vector<Sequence> Simulate(IProcess& process, const double& T) = 0;

	virtual std::vector<Sequence> Simulate(IProcess& process, const unsigned& n) = 0;

};