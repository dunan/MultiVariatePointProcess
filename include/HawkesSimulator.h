#include <vector>
#include <Process.h>
#include <Sequence.h>
#include <HawkesProcess.h>

class HawkesSimulator : public Simulator
{

private:

	HawkesProcess hawkes_;

public:

	HawkesSimulator(const HawkesProcess& process) : hawkes_(process){}

	~HawkesSimulator(){}

};