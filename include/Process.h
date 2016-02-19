#include <vector>
#include <Sequence.h>

class IProcess
{

private:

	std::vector<double> parameters_;

	unsigned num_dims_;

public:

	IProcess(const unsigned& n, const unsigned& num_dims)
	{
		parameters_ = std::vector<double>(n, 0);

		num_dims_ = num_dims;
	}

	~IProcess(){}

	std::vector<double> GetParameters() {return parameters_;}

	void SetParameters(std::vector<double> v) {parameters_ = v;}

	virtual double Intensity(const double& t) = 0;

	virtual Loglikelihood(const std::vector<Sequence>& data, double& objvalue, std::vector<double>& Gradient) = 0;

	virtual double IntensityUpperBound(const double& t, const Sequence& data) = 0;

};