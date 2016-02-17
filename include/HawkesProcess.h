#include <vector>
#include <Sequence.h>
#include <Process.h>

class HawkesProcess : public Process
{

private:

	std::vector<std::vector<double> > data_;

public:

	HawkesProcess(const unsigned& num_dims) {}

	~HawkesProcess(){}

	double Intensity(const double& t) {}

	double Loglikelihood(const std::vector<Sequence>& data) {}

	std::vector<double> Gradient() {}

	double IntensityUpperBound(const double& t) {}

};