#include <vector>
#include <Process.h>
#include <Sequence.h>

class UniversalSimulator : public Simulator
{

private:

	Process process_;

public:

	UniversalSimulator(const Process& process) : process_(process){}

	~UniversalSimulator(){}

};