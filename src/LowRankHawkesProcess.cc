/**
 * \file LowRankHawkesProcess.cc
 * \brief The class implementation of LowRankHawkesProcess implementing the low-rank Hawkes process.
 */
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
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

	options_ = options;

	Optimizer opt(this);

	opt.ProximalFrankWolfeForLowRankHawkes(options_.ini_learning_rate, options_.coefficients[LAMBDA0], options_.coefficients[LAMBDA], options_.ub_nuclear_lambda0, options_.ub_nuclear_alpha, options_.rho, options_.ini_max_iter, num_rows_, num_cols_);

}

void LowRankHawkesProcess::fit(const std::vector<Sequence>& data, const LowRankHawkesProcess::OPTION& options, const Eigen::MatrixXd& TrueLambda0, const Eigen::MatrixXd& TrueAlpha)
{

	Initialize(data);

	options_ = options;

	Optimizer opt(this);

	opt.ProximalFrankWolfeForLowRankHawkes(options_.ini_learning_rate, options_.coefficients[LAMBDA0], options_.coefficients[LAMBDA], options_.ub_nuclear_lambda0, options_.ub_nuclear_alpha, options_.rho, options_.ini_max_iter, num_rows_, num_cols_, TrueLambda0, TrueAlpha);
}

double LowRankHawkesProcess::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	return 0;
}

double LowRankHawkesProcess::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	return 0;
}

void LowRankHawkesProcess::ExpectationHandler::operator()(const Eigen::VectorXd& t, Eigen::VectorXd& y)
{
	Eigen::Map<Eigen::MatrixXd> Lambda0 = Eigen::Map<Eigen::MatrixXd>(parent_.parameters_.segment(0, parent_.num_dims_).data(), parent_.num_rows_, parent_.num_cols_);
	
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(parent_.parameters_.segment(parent_.num_dims_, parent_.num_dims_).data(), parent_.num_rows_, parent_.num_cols_);

	Eigen::Map<Eigen::MatrixXd> Beta = Eigen::Map<Eigen::MatrixXd>(parent_.beta_.segment(0, parent_.num_dims_).data(), parent_.num_rows_, parent_.num_cols_);

	std::vector<Event> events = sequence_.GetEvents();
	
	unsigned num_elements = t.size();
	y = Eigen::VectorXd::Zero(num_elements);
	Eigen::VectorXd expsum = Eigen::VectorXd::Zero(events.size());

	for(unsigned i = 1; i < events.size(); ++ i)
	{
		expsum(i) = exp(-Beta(uid_, itemid_) * (events[i].time - events[i-1].time)) * (1 + expsum(i - 1));
	}

	y = (Lambda0(uid_, itemid_) + A(uid_, itemid_) * (1 + expsum(expsum.size() - 1)) * (-Beta(uid_, itemid_) * t.array()).exp()) * (-Lambda0(uid_, itemid_) * t.array() - (A(uid_, itemid_) / Beta(uid_, itemid_)) * (1 + expsum(expsum.size() - 1)) * (1 - (-Beta(uid_, itemid_) * t.array()).exp())).exp() * t.array();

}

double LowRankHawkesProcess::PredictNextEventTime(unsigned uid, unsigned itemid, double T, const std::vector<Sequence>& data)
{

	assert((uid >= 0) && (uid < num_rows_) && (itemid >= 0) && (itemid < num_cols_));
	unsigned i, j;
	for(unsigned c = 0; c < data.size(); ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();
		unsigned dimID = seq[0].DimentionID;
		Ind2Vec(dimID, i, j);

		if((i == uid) && (j == itemid))
		{
			ExpectationHandler handler(uid, itemid, data[c], *this);

			return seq[seq.size() - 1].time + SimpsonIntegral38(handler, 0, 100, 9000);
		}
	}

	Eigen::Map<Eigen::MatrixXd> Lambda0 = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(0, num_dims_).data(), num_rows_, num_cols_);

	return std::min(1 / Lambda0(uid, itemid), T);
}

unsigned LowRankHawkesProcess::PredictNextItem(unsigned uid, double t, const std::vector<Sequence>& data)
{
	assert((uid >= 0) && (uid < num_rows_));
	unsigned i, j;
	std::map<unsigned, std::map<unsigned, unsigned> > user_item_pair;

	for(unsigned c = 0; c < data.size(); ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();
		unsigned dimID = seq[0].DimentionID;
		Ind2Vec(dimID, i, j);

		if(i == uid)
		{
			if(user_item_pair.find(i) == user_item_pair.end())
			{
				std::map<unsigned, unsigned> item_to_index;
				item_to_index.insert(std::make_pair(j, c));
				user_item_pair.insert(std::make_pair(i, item_to_index));
			}else if(user_item_pair[i].find(j) == user_item_pair[i].end())
			{
				user_item_pair[i].insert(std::make_pair(j, c));
			}
		}
	}

	Eigen::Map<Eigen::MatrixXd> Lambda0 = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(0, num_dims_).data(), num_rows_, num_cols_);
	
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_).data(), num_rows_, num_cols_);

	Eigen::Map<Eigen::MatrixXd> Beta = Eigen::Map<Eigen::MatrixXd>(beta_.segment(0, num_dims_).data(), num_rows_, num_cols_);

	std::vector<std::pair<double, unsigned> > intensity_list;

	if(user_item_pair.size() == 0)
	{
		for(unsigned j = 0; j < num_cols_; ++ j)
		{
			intensity_list.push_back(std::make_pair(Lambda0(uid, j), j));
		}
	}else
	{
		for(unsigned j = 0; j < num_cols_; ++ j)
		{
			if(user_item_pair[uid].find(j) == user_item_pair[uid].end())
			{
				intensity_list.push_back(std::make_pair(Lambda0(uid, j), j));
			}else
			{
				double intensity = Lambda0(uid, j);
				const std::vector<Event>& events = data[user_item_pair[uid][j]].GetEvents();
				for(unsigned k = 0; k < events.size(); ++ k)
				{
					if(events[k].time < t)
					{
						intensity += A(uid, j) * exp(-Beta(uid, j) * (t - events[k].time));
					}
				}
				intensity_list.push_back(std::make_pair(intensity, j));
			}
		}
	}
	
	std::sort(intensity_list.begin(), intensity_list.end(), std::greater<std::pair<double, unsigned> >());

	return intensity_list[0].second;
}
