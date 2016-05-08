/**
 * \file HawkesGeneralKernel.cc
 * \brief The class implementation of HawkesGeneralKernel for [Hawkes](@ref PlainHawkes) process with customized triggering kernels.
 */
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include "../include/HawkesGeneralKernel.h"
#include "../include/Sequence.h"
#include "../include/Optimizer.h"
#include "../include/OgataThinning.h"
#include "../include/Utility.h"
#include "../include/GNUPlotWrapper.h"
#include "../include/SimpleRNG.h"

void HawkesGeneralKernel::Initialize(const std::vector<Sequence>& data)
{
	num_sequences_ = data.size();

	arrayK = std::vector<std::vector<Eigen::MatrixXd> >(num_dims_, std::vector<Eigen::MatrixXd>(num_sequences_, Eigen::MatrixXd()));

	arrayG = std::vector<Eigen::MatrixXd>(num_dims_, Eigen::MatrixXd());

	InitializeDimension(data);

	observation_window_T_ = Eigen::VectorXd::Zero(num_sequences_);

	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		observation_window_T_(c) = data[c].GetTimeWindow();
	}

	for(unsigned n = 0; n < num_dims_; ++ n)
	{
		Eigen::MatrixXd& MatrixG = arrayG[n];
		MatrixG = Eigen::MatrixXd::Zero(num_sequences_, num_dims_);

		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			Eigen::MatrixXd& MatrixK = arrayK[n][c];
			MatrixK = Eigen::MatrixXd::Zero(all_timestamp_per_dimension_[c][n].size(), num_dims_);

			for (unsigned i = 0; i < all_timestamp_per_dimension_[c][n].size(); ++i)
			{
				for(unsigned m = 0; m < num_dims_; ++ m)
				{
					for(unsigned j = 0; j < all_timestamp_per_dimension_[c][m].size(); ++ j)
					{
						if(all_timestamp_per_dimension_[c][m][j] < all_timestamp_per_dimension_[c][n][i])
						{
							MatrixK(i, m) += (*triggeringkernels_[m][n])(all_timestamp_per_dimension_[c][n][i] - all_timestamp_per_dimension_[c][m][j]);
						}
					}
				}
			}

			for(unsigned m = 0; m < num_dims_; ++ m)
			{
				for(unsigned j = 0; j < all_timestamp_per_dimension_[c][m].size(); ++ j)
				{
					MatrixG(c, m) += triggeringkernels_[m][n]->Integral(0, observation_window_T_(c) - all_timestamp_per_dimension_[c][m][j]);
				}
			}
		}
	}
}

//  MLE esitmation of the parameters
void HawkesGeneralKernel::fit(const std::vector<Sequence>& data, const OPTION& options)
{
	HawkesGeneralKernel::Initialize(data);

	options_ = options;

	Optimizer opt(this);
	opt.PLBFGS(0, 1e10);

	RestoreOptionToDefault();

}

void HawkesGeneralKernel::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	gradient = Eigen::VectorXd::Zero(num_dims_ * (1 + num_dims_));

	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> grad_alpha_matrix = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	objvalue = 0;

	for(unsigned n = 0; n < num_dims_; ++ n)
	{
		double obj_n = 0;

		Eigen::MatrixXd& MatrixG = arrayG[n];

		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			Eigen::MatrixXd& MatrixK = arrayK[n][c];

			Eigen::VectorXd intensity = Lambda0_(n) + (MatrixK * Alpha_.col(n)).transpose().array() + 1e-4;

			for (unsigned i = 0; i < all_timestamp_per_dimension_[c][n].size(); ++i)
			{
				grad_alpha_matrix.col(n) = grad_alpha_matrix.col(n).array() + MatrixK.row(i).transpose().array() / intensity(i);
				grad_lambda0_vector(n) = grad_lambda0_vector(n) + 1 / intensity(i);
				obj_n += log(intensity(i));
			}

			grad_alpha_matrix.col(n) = grad_alpha_matrix.col(n) - MatrixG.row(c).transpose();
			grad_lambda0_vector(n) = grad_lambda0_vector(n) - observation_window_T_(c);

			obj_n = obj_n - Lambda0_(n) * observation_window_T_(c) - MatrixG.row(c) * Alpha_.col(n);
		}

		objvalue += obj_n;
	}

	objvalue = - objvalue / num_sequences_;

	gradient = - gradient.array() / num_sequences_;


	// Regularization for base intensity
	switch (options_.base_intensity_regularizer)
	{
		case L22 :
			
			grad_lambda0_vector = grad_lambda0_vector.array() + (options_.coefficients[LAMBDA0] * Lambda0_.array());

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA0] * Lambda0_.squaredNorm();
			
			break;

		case L1 :

			grad_lambda0_vector = grad_lambda0_vector.array() + options_.coefficients[LAMBDA0];

			objvalue = objvalue + options_.coefficients[LAMBDA0] * Lambda0_.array().abs().sum();

			break;

		default:
			break; 	
	}

	// Regularization for excitation matrix
	Eigen::Map<Eigen::VectorXd> grad_alpha_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_ * num_dims_);
	Eigen::Map<Eigen::VectorXd> alpha_vector = Eigen::Map<Eigen::VectorXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_ * num_dims_);
	switch (options_.excitation_regularizer)
	{
		case L22 :

			grad_alpha_vector = grad_alpha_vector.array() + (options_.coefficients[LAMBDA] * alpha_vector.array());

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA] * alpha_vector.squaredNorm();

			break;

		case L1 :

			grad_alpha_vector = grad_alpha_vector.array() + options_.coefficients[LAMBDA];

			objvalue = objvalue + options_.coefficients[LAMBDA] * alpha_vector.array().abs().sum();

			break;

		default:
			break;
	}

}

void HawkesGeneralKernel::Gradient(const unsigned &k, Eigen::VectorXd& gradient)
{

}

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
double HawkesGeneralKernel::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
{
	intensity_dim = Eigen::VectorXd::Zero(num_dims_);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	intensity_dim = Lambda0_;

	const std::vector<Event>& seq = data.GetEvents();

	for(unsigned i = 0; i < seq.size(); ++ i)
	{
		if (seq[i].time < t)
		{
			for(unsigned d = 0; d < num_dims_; ++ d)
			{
				intensity_dim(d) += Alpha_(seq[i].DimentionID, d) * (*triggeringkernels_[seq[i].DimentionID][d])(t - seq[i].time);
			}	
		}
		else
		{
			break;
		}
	}

	return intensity_dim.array().sum();
}

//  Thisfunction requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
double HawkesGeneralKernel::IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
{
	intensity_upper_dim = Eigen::VectorXd::Zero(num_dims_);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	intensity_upper_dim = Lambda0_;

	const std::vector<Event>& seq = data.GetEvents();

	for(unsigned i = 0; i < seq.size(); ++ i)
	{
		if (seq[i].time <= t)
		{
			for(unsigned d = 0; d < num_dims_; ++ d)
			{
				intensity_upper_dim(d) += Alpha_(seq[i].DimentionID, d) * (*triggeringkernels_[seq[i].DimentionID][d])(t - seq[i].time);
			}	
		}
		else
		{
			break;
		}
	}

	return intensity_upper_dim.array().sum();
}

//  Thisfunction requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the integral of the intensity from a to b
double HawkesGeneralKernel::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{

	std::vector<Sequence> sequences;
	sequences.push_back(data);

	InitializeDimension(sequences);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	std::vector<std::vector<double> >& timestamp_per_dimension = all_timestamp_per_dimension_[0];

	Eigen::VectorXd integral_value = Eigen::VectorXd::Zero(num_dims_);

	for(unsigned n = 0; n < num_dims_; ++ n)
	{
		integral_value(n) = Lambda0_(n) * (upper - lower);

		for(unsigned m = 0; m < num_dims_; ++ m)
		{
			for(unsigned i = 0; i < timestamp_per_dimension[m].size(); ++ i)
			{
				if(timestamp_per_dimension[m][i] <= lower)
				{
					integral_value(n) += Alpha_(m,n) * triggeringkernels_[m][n]->Integral(lower - timestamp_per_dimension[m][i], upper - timestamp_per_dimension[m][i]);
				}else if ((timestamp_per_dimension[m][i] > lower) && (timestamp_per_dimension[m][i] <= upper))
				{
					integral_value(n) += Alpha_(m,n) * triggeringkernels_[m][n]->Integral(0, upper - timestamp_per_dimension[m][i]);
				}else
				{
					break;
				}
			}

		}
	}

	return integral_value.sum();
}

//  This function predicts the next event by simulation;
double HawkesGeneralKernel::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	OgataThinning ot(num_dims_);
	double t = 0;
	for(unsigned i = 0; i < num_simulations; ++ i)
	{
		Event event = ot.SimulateNext(*this, data);
		t += event.time;
	}
	return t / num_simulations;
}

void HawkesGeneralKernel::RestoreOptionToDefault()
{
	options_.method = PLBFGS;
	options_.base_intensity_regularizer = NONE;
	options_.excitation_regularizer = NONE;
	options_.coefficients[LAMBDA0] = 0;
	options_.coefficients[LAMBDA] = 0;
}