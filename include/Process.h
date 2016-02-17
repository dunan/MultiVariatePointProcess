#include <vector>
#include <Sequence.h>

class Process
{

private:

	std::vector<double> parameters_;

	unsigned num_dims_;

public:

	Process(const unsigned& n, const unsigned& num_dims)
	{
		parameters_ = std::vector<double>(n, 0);

		num_dims_ = num_dims;
	}

	~Process(){}

	std::vector<double> GetParameters() {return parameters_;}

	void SetParameters(std::vector<double> v) {parameters_ = v;}

	virtual double Intensity(const double& t) = 0;

	virtual double Loglikelihood(const std::vector<Sequence>& data) = 0;

	virtual std::vector<double> Gradient() = 0;

	virtual double IntensityUpperBound(const double& t) = 0;

};