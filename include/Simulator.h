#include <vector>
#include <Process.h>
#include <Sequence.h>

class Simulator
{

public:

	virtual std::vector<Sequence> Simulate(const double& T) = 0;

	virtual std::vector<Sequence> Simulate(const unsigned& n) = 0;

};