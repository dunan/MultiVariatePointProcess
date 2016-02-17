#include <vector>
#include <Process.h>
#include <Sequence.h>

class UniversalLearner : public Learner
{

private:

	Process process_;

public:

	UniversalLearner(const Process& process) : process_(process){}

	~UniversalLearner(){}

	std::vector<double> fit(std::vector<Sequence>& data, const double& T){}
};