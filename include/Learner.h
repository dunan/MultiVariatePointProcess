#include <vector>
#include <Process.h>
#include <Sequence.h>

class Learner
{
public:

	virtual std::vector<double> fit(std::vector<Sequence>& data, const double& T) = 0;
};