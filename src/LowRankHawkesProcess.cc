#include <vector>
#include <cmath>
#include <iostream>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include "../include/LowRankHawkesProcess.h"
#include "../include/Sequence.h"
#include "../include/Optimizer.h"
#include "../include/OgataThinning.h"


unsigned LowRankHawkesProcess::Vec2Ind(const unsigned& i, const unsigned& j)
{
	return num_rows_ * i + j;
}

void LowRankHawkesProcess::Ind2Vec(const unsigned& ind, unsigned& i, unsigned& j)
{
	j = ind / num_rows_;

	i = ind % num_rows_;
}

void LowRankHawkesProcess::Initialize(const std::vector<Sequence>& data)
{
	observed_idx_ = Eigen::VectorXi::Zero(data.size());

	unsigned num_total_events = 0;
	unsigned num_total_pairs = data.size();
	for(unsigned c = 0; c < num_total_pairs; ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();
		num_total_events += seq.size();
	}

	event_intensity_features_ = Eigen::VectorXd::Zero(num_total_events);

	integral_intensity_features_ = Eigen::VectorXd::Zero(num_total_pairs);

	observation_window_T_ = Eigen::VectorXd::Zero(num_total_pairs);

	Eigen::VectorXi pairidx = Eigen::VectorXi::Zero(num_total_events);

	unsigned current_pos = 0;
	for(unsigned c = 0; c < num_total_pairs; ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();
		unsigned dimID = seq[0].DimentionID;

		observed_idx_(c) = dimID;

		Eigen::Map<Eigen::VectorXd> current_event_intensity_feature = Eigen::Map<Eigen::VectorXd>(event_intensity_features_.segment(current_pos, seq.size()).data(), seq.size());

		double Tc = data[c].GetTimeWindow();
		
		observation_window_T_(c) = Tc;

		integral_intensity_features_(c) = (1 - exp(-beta_(dimID) * (Tc - seq[0].time)));
		for(unsigned i = 1; i < seq.size(); ++ i)
		{
			current_event_intensity_feature(i) = exp(-beta_(dimID) * (seq[i].time - seq[i - 1].time)) * (1 + current_event_intensity_feature(i - 1));
			integral_intensity_features_(c) = integral_intensity_features_(c) + (1 - exp(-beta_(dimID) * (Tc - seq[i].time)));
		}

		integral_intensity_features_(c) = (1 / beta_(dimID)) * integral_intensity_features_(c);

		pairidx.segment(current_pos, seq.size()) = Eigen::VectorXi::Constant(seq.size(), c);

		current_pos += seq.size();
	}

	std::vector<Eigen::Triplet<double> > triplet_list(num_total_events, Eigen::Triplet<double>());

	for(unsigned i = 0; i < num_total_events; ++ i)
	{
		triplet_list[i] = Eigen::Triplet<double>(pairidx(i), i, 1.0);
	}

	pair_event_map_ = Eigen::SparseMatrix<double>(num_total_pairs, num_total_events);
	pair_event_map_.setFromTriplets(triplet_list.begin(), triplet_list.end());

}


double LowRankHawkesProcess::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
{
	return 0;
}

double LowRankHawkesProcess::IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
{
	return 0;
}

void LowRankHawkesProcess::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	if(observed_idx_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	gradient = Eigen::VectorXd::Zero(2 * num_dims_);

	Eigen::VectorXd Lambda0 = parameters_.segment(0, num_dims_);
	Eigen::VectorXd ObservedLambda0;
	igl::slice(Lambda0, observed_idx_, ObservedLambda0);

	Eigen::VectorXd Alpha = parameters_.segment(num_dims_, num_dims_);
	Eigen::VectorXd ObservedAlpha;
	igl::slice(Alpha, observed_idx_, ObservedAlpha);

	Eigen::VectorXd intensity = ((ObservedLambda0.transpose() * pair_event_map_).array() + ((ObservedAlpha.transpose() * pair_event_map_).array() * event_intensity_features_.transpose().array()).array()).transpose();

	Eigen::VectorXd GradObservedLambda0 = (pair_event_map_ * (1 / intensity.array()).matrix()) - observation_window_T_;
	igl::slice_into(GradObservedLambda0, observed_idx_, gradient);

	Eigen::VectorXd GradObservedAlpha = pair_event_map_ * (event_intensity_features_.array() / intensity.array()).matrix() - integral_intensity_features_;
	igl::slice_into(GradObservedAlpha, observed_idx_.array() + num_dims_, gradient);

	gradient = -gradient.array() / double(observed_idx_.size());

	objvalue = - ((pair_event_map_ * (intensity.array().log()).matrix()).sum() - observation_window_T_.transpose() * ObservedLambda0 - integral_intensity_features_.transpose() * ObservedAlpha) / double(observed_idx_.size());
}

void LowRankHawkesProcess::Gradient(const unsigned &k, Eigen::VectorXd& gradient)
{

}

void LowRankHawkesProcess::fit(const std::vector<Sequence>& data, const LowRankHawkesProcess::OPTION& options)
{
	Initialize(data);

}

void LowRankHawkesProcess::fit(const std::vector<Sequence>& data, const LowRankHawkesProcess::OPTION& options, const Eigen::MatrixXd& TrueLambda0, const Eigen::MatrixXd& TrueAlpha)
{

	Initialize(data);

	options_ = options;

	Optimizer opt(this);

	opt.ProximalFrankWolfeForLowRankHawkes(1e-2, 1.0, 1.0, 25, 25, 1e1, 1000, 64, 64, TrueLambda0, TrueAlpha);
}

double LowRankHawkesProcess::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	return 0;
}

double LowRankHawkesProcess::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	return 0;
}
