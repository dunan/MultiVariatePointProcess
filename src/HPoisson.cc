#include <vector>
#include <cmath>
#include "../include/HPoisson.h"
#include "../include/Sequence.h"

double HPoisson::Intensity(const double& t, const Sequence& data, std::vector<double>& intensity_dim)
{

	intensity_dim = std::vector<double>();

	double sum = 0;

	const std::vector<double>& parameters = IProcess::GetParameters();

	for(std::vector<double>::const_iterator i_param = parameters.begin(); i_param != parameters.end(); ++ i_param)
	{
		sum += *i_param;
		intensity_dim.push_back(*i_param);
	}

	return sum;
}

double HPoisson::IntensityUpperBound(const double& t, const Sequence& data, std::vector<double>& intensity_upper_dim)
{
	return HPoisson::Intensity(t, data, intensity_upper_dim);
}

void HPoisson::Initialize(const std::vector<Sequence>& data)
{
	num_sequences_ = data.size();

	const unsigned& D = IProcess::GetNumDims();

	std::vector<unsigned> event_number_by_dim(D, 0);

	intensity_itegral_features_ = 0;

	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();

		for(unsigned i = 0; i < seq.size(); ++ i)
		{
			++ event_number_by_dim[seq[i].DimentionID];
		}

		intensity_itegral_features_ += data[c].GetTimeWindow();

	}

	intensity_itegral_features_ /= double(num_sequences_);

	intensity_features_ = std::vector<double>(D, 0);

	for(unsigned d = 0; d < D; ++ d)
	{
		intensity_features_[d] = double(event_number_by_dim[d]) / double(num_sequences_);
	}

}

void HPoisson::NegLoglikelihood(double& objvalue, std::vector<double>& gradient)
{
	
	const unsigned& D = IProcess::GetNumDims();
	
	objvalue = 0;

	gradient = std::vector<double>(D, 0);

	const std::vector<double>& params = IProcess::GetParameters();

	for(unsigned d = 0; d < D; ++ d)
	{
		objvalue += (intensity_features_[d] * log(params[d]) - intensity_itegral_features_ * params[d]);
		gradient[d] =  intensity_features_[d] / params[d] - intensity_itegral_features_;
	}	

	for(unsigned d = 0; d < D; ++ d)
	{
		gradient[d] =  -gradient[d];
	}

	objvalue = -objvalue;

}

void HPoisson::Gradient(const unsigned &k, std::vector<double>& gradient)
{
	return;
}


