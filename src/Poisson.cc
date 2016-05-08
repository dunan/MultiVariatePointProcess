/**
 * \file Poisson.cc
 * \brief The class implementation of Poisson implementing the homogeneous Poisson process.
 */
#include <vector>
#include <cmath>
#include "../include/Poisson.h"
#include "../include/Sequence.h"

double Poisson::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
{
	const Eigen::VectorXd& parameters = IProcess::GetParameters();

	intensity_dim = Eigen::VectorXd(parameters);

	return parameters.array().sum();
}

double Poisson::IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
{
	return Poisson::Intensity(t, data, intensity_upper_dim);
}

void Poisson::Initialize(const std::vector<Sequence>& data)
{
	num_sequences_ = data.size();

	const unsigned& D = IProcess::GetNumDims();

	InitializeDimension(data);

	// all_timestamp_per_dimension_ = std::vector<std::vector<std::vector<double> > >(num_sequences_, std::vector<std::vector<double> > (D, std::vector<double> ()));

	Eigen::VectorXd event_number_by_dim = Eigen::VectorXd::Zero(D);

	observation_window_T_ = Eigen::VectorXd::Zero(num_sequences_);

	intensity_itegral_features_ = 0;

	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();

		for(unsigned i = 0; i < seq.size(); ++ i)
		{
			++ event_number_by_dim(seq[i].DimentionID);
			// all_timestamp_per_dimension_[c][seq[i].DimentionID].push_back(seq[i].time);
		}

		intensity_itegral_features_ += data[c].GetTimeWindow();

		observation_window_T_(c) = data[c].GetTimeWindow();
	}

	intensity_itegral_features_ /= double(num_sequences_);

	intensity_features_ = event_number_by_dim.array() / double(num_sequences_);
}

void Poisson::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	
	const unsigned& D = IProcess::GetNumDims();

	gradient = Eigen::VectorXd::Zero(D);

	const Eigen::VectorXd& params = IProcess::GetParameters();

	objvalue = - (intensity_features_.array() * params.array().log() - intensity_itegral_features_ * params.array()).sum();

	gradient = - (intensity_features_.array() / params.array() - intensity_itegral_features_);

}

void Poisson::Gradient(const unsigned &c, Eigen::VectorXd& gradient)
{
	const unsigned& D = IProcess::GetNumDims();

	const Eigen::VectorXd& params = IProcess::GetParameters();

	Eigen::VectorXd event_number_by_dim = Eigen::VectorXd::Zero(D);

	for(unsigned d = 0; d < D; ++ d)
	{
		event_number_by_dim(d) = all_timestamp_per_dimension_[c][d].size();
	}

	gradient = - (event_number_by_dim.array() / params.array() - observation_window_T_(c));
	
}

void Poisson::fit(const std::vector<Sequence>& data)
{
	Initialize(data);

	IProcess::SetParameters(intensity_features_.array() / intensity_itegral_features_);
}

double Poisson::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	
	return 0;
}

double Poisson::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	return 0;
}
