/**
 * \file SelfInhibitingProcess.cc
 * \brief The class implementation of SelfInhibitingProcess implementing the standard Self-Inhibiting (or Self-Correcting) process.
 */
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <sstream>
#include "../include/SelfInhibitingProcess.h"
#include "../include/Sequence.h"
#include "../include/Optimizer.h"
#include "../include/OgataThinning.h"
#include "../include/Utility.h"
#include "../include/GNUPlotWrapper.h"

void SelfInhibitingProcess::Initialize(const std::vector<Sequence>& data)
{
	num_sequences_ = data.size();

	arrayK_ = std::vector<std::vector<Eigen::MatrixXd > >(num_sequences_, std::vector<Eigen::MatrixXd >(num_dims_, Eigen::MatrixXd()));

	arrayG_ = std::vector<std::vector<std::vector<std::vector<std::pair<double, unsigned> > > > >(num_sequences_, std::vector<std::vector<std::vector<std::pair<double, unsigned> > > >(num_dims_, std::vector<std::vector<std::pair<double, unsigned> > >()));

	InitializeDimension(data);

	observation_window_T_ = Eigen::VectorXd::Zero(num_sequences_);

	for(unsigned k = 0; k < num_sequences_; ++ k)
	{
		observation_window_T_(k) = data[k].GetTimeWindow();

		for(unsigned n = 0; n < num_dims_; ++ n)
		{
			arrayK_[k][n] = Eigen::MatrixXd::Zero(all_timestamp_per_dimension_[k][n].size(), num_dims_);

			Eigen::MatrixXd& influence_to_event_i = arrayK_[k][n];

			for(unsigned i = 0; i < all_timestamp_per_dimension_[k][n].size(); ++ i)
			{
				for(unsigned m = 0; m < num_dims_; ++ m)
				{
					for(unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size(); ++ j)
					{
						if(all_timestamp_per_dimension_[k][m][j] < all_timestamp_per_dimension_[k][n][i])
						{
							++ influence_to_event_i(i, m);
						}else
						{
							break;
						}
					}
				}
			}

			// caclulate events between current i and previous i - 1 event for dimension n in the sequence k

			arrayG_[k][n] = std::vector<std::vector<std::pair<double, unsigned> > >(all_timestamp_per_dimension_[k][n].size() + 1, std::vector<std::pair<double, unsigned> >());

			// handle the first event since we need to start from time 0
			std::vector<std::pair<double, unsigned> >& events_between_0_and_first_event = arrayG_[k][n][0];
			for(unsigned m = 0; m < num_dims_; ++ m)
			{
				for(unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size(); ++ j)
				{
					if(all_timestamp_per_dimension_[k][m][j] <= all_timestamp_per_dimension_[k][n][0])
					{
						events_between_0_and_first_event.push_back(std::make_pair(all_timestamp_per_dimension_[k][m][j], m));
					}else
					{
						break;
					}
				}
			}
			events_between_0_and_first_event.push_back(std::make_pair(0, n));
			sort(events_between_0_and_first_event.begin(), events_between_0_and_first_event.end());

			// handle the rest events
			for(unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size(); ++ i)
			{
				std::vector<std::pair<double, unsigned> >& events_between_prev_and_current_i = arrayG_[k][n][i];

				for(unsigned m = 0; m < num_dims_; ++ m)
				{
					for(unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size(); ++ j)
					{
						if((all_timestamp_per_dimension_[k][m][j] <= all_timestamp_per_dimension_[k][n][i]) && (all_timestamp_per_dimension_[k][m][j] >= all_timestamp_per_dimension_[k][n][i - 1]))
						{
							events_between_prev_and_current_i.push_back(std::make_pair(all_timestamp_per_dimension_[k][m][j], m));
						}else if (all_timestamp_per_dimension_[k][m][j] > all_timestamp_per_dimension_[k][n][i])
						{
							break;
						}
					}
				}

				sort(events_between_prev_and_current_i.begin(), events_between_prev_and_current_i.end());

			}

			// handle the events on the other dimensions between the last event of dimension n and T in the sequence k
			unsigned i = all_timestamp_per_dimension_[k][n].size();
			std::vector<std::pair<double, unsigned> >& events_between_last_and_T = arrayG_[k][n][i];
			for(unsigned m = 0; m < num_dims_; ++ m)
			{
				for(unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size(); ++ j)
				{
					if((all_timestamp_per_dimension_[k][m][j] < observation_window_T_(k)) && (all_timestamp_per_dimension_[k][m][j] >= all_timestamp_per_dimension_[k][n][i - 1]))
					{
						events_between_last_and_T.push_back(std::make_pair(all_timestamp_per_dimension_[k][m][j], m));
					}
				}
			}
			events_between_last_and_T.push_back(std::make_pair(observation_window_T_(k), n));
			sort(events_between_last_and_T.begin(), events_between_last_and_T.end());
	
		}
	}

}


double SelfInhibitingProcess::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
{
	intensity_dim = Eigen::VectorXd::Zero(num_dims_);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Beta_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	const std::vector<Event>& seq = data.GetEvents();

	intensity_dim = Lambda0_.array() * t;

	for(unsigned i = 0; i < seq.size(); ++ i)
	{
		if(seq[i].time < t)
		{
			for(unsigned d = 0; d < num_dims_; ++ d)
			{
				intensity_dim(d) = intensity_dim(d) - Beta_(seq[i].DimentionID, d);
			}
		}else
		{
			break;
		}
	}

	intensity_dim = intensity_dim.array().exp();

	return intensity_dim.array().sum();
}

double SelfInhibitingProcess::IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
{
	intensity_upper_dim = Eigen::VectorXd::Zero(num_dims_);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Beta_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	const std::vector<Event>& seq = data.GetEvents();

	intensity_upper_dim = Lambda0_.array() * (t + L);

	for(unsigned i = 0; i < seq.size(); ++ i)
	{
		if(seq[i].time <= t)
		{
			for(unsigned d = 0; d < num_dims_; ++ d)
			{
				intensity_upper_dim(d) = intensity_upper_dim(d) - Beta_(seq[i].DimentionID, d);
			}
		}else
		{
			break;
		}
	}

	intensity_upper_dim = intensity_upper_dim.array().exp();

	return intensity_upper_dim.array().sum();
}

void SelfInhibitingProcess::GetNegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	gradient = Eigen::VectorXd::Zero(num_dims_ * (1 + num_dims_));

	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> grad_beta_matrix = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Beta_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	objvalue = 0;

	for (unsigned k = 0; k < num_sequences_; ++k) 
	{

		const std::vector<std::vector<double> > &timestamp_per_dimension = all_timestamp_per_dimension_[k];

		for (unsigned n = 0; n < num_dims_; ++n) 
	    {

			double obj_n = 0;

			for (unsigned i = 0; i < timestamp_per_dimension[n].size(); ++i) 
			{
				const double& t_n_i = timestamp_per_dimension[n][i];

				obj_n += (Lambda0_(n) * t_n_i - arrayK_[k][n].row(i) * Beta_.col(n));

				grad_lambda0_vector(n) = grad_lambda0_vector(n) + t_n_i;

				grad_beta_matrix.col(n) = grad_beta_matrix.col(n) - arrayK_[k][n].row(i).transpose();

			}

			for (unsigned i = 0; i <= timestamp_per_dimension[n].size(); ++i) 
			{
				// integral by segment
				double past_influence_to_pre_i = (i == 0 ? 0.0 : arrayK_[k][n].row(i - 1) * Beta_.col(n));

				for(unsigned b = 1; b < arrayG_[k][n][i].size(); ++ b)
				{
					const double& t_m_b = arrayG_[k][n][i][b].first;

					const double& t_m_pre_b = arrayG_[k][n][i][b - 1].first;

					double past_influence_from_pre_i = 0;

					Eigen::VectorXd beta_count = Eigen::VectorXd::Zero(num_dims_);

					for(unsigned p = 0; p < b; ++ p)
					{
						if(arrayG_[k][n][i][p].first != 0)
						{
							past_influence_from_pre_i += Beta_(arrayG_[k][n][i][p].second, n);
							beta_count(arrayG_[k][n][i][p].second) = beta_count(arrayG_[k][n][i][p].second) + 1;
						}
					}

					double part_a = t_m_b * Lambda0_(n) - past_influence_to_pre_i - past_influence_from_pre_i;
					double part_b = t_m_pre_b * Lambda0_(n) - past_influence_to_pre_i - past_influence_from_pre_i;

					obj_n -= (exp(part_a) - exp(part_b)) / Lambda0_(n);

					grad_lambda0_vector(n) = grad_lambda0_vector(n) - (exp(part_a) * (t_m_b * Lambda0_(n) - 1) - exp(part_b) * (t_m_pre_b * Lambda0_(n) - 1)) / (Lambda0_(n) * Lambda0_(n));

					if(i > 0)
					{
						grad_beta_matrix.col(n) = grad_beta_matrix.col(n).array() - (exp(part_a) - exp(part_b)) * (-arrayK_[k][n].row(i - 1).transpose().array() - beta_count.array()) / Lambda0_(n);
					}
				}
			}

			objvalue += obj_n;

	  	}
	}

	objvalue = -objvalue / num_sequences_;

	gradient = - gradient.array() / num_sequences_;	
}

void SelfInhibitingProcess::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	GetNegLoglikelihood(objvalue, gradient);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);
	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims_).data(), num_dims_);

	// Regularization for base intensity
	switch (options_.base_intensity_regularizer)
	{
		case L22 :
			
			grad_lambda0_vector = grad_lambda0_vector.array() + (options_.coefficients[LAMBDA0] * Lambda0_.array());

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA0] * Lambda0_.squaredNorm();
			
			break;

		case NONE :

			break;

		default:
			break; 	
	}

	// Regularization for excitation matrix
	Eigen::Map<Eigen::VectorXd> grad_beta_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_ * num_dims_);

	Eigen::Map<Eigen::VectorXd> beta_vector = Eigen::Map<Eigen::VectorXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_ * num_dims_);

	// Regularization for excitation matrix
	switch (options_.excitation_regularizer)
	{
		case L1 :

			grad_beta_vector = grad_beta_vector.array() + options_.coefficients[LAMBDA];

			objvalue += options_.coefficients[LAMBDA] * beta_vector.array().abs().sum();

			return;

		case L22 :

			grad_beta_vector = grad_beta_vector.array() + options_.coefficients[LAMBDA] * beta_vector.array();

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA] * beta_vector.squaredNorm();

			return;

		case NONE :
			return;
	}


}

void SelfInhibitingProcess::Gradient(const unsigned &k, Eigen::VectorXd& gradient)
{
	
}

void SelfInhibitingProcess::fit(const std::vector<Sequence>& data, const OPTION& options)
{
	Initialize(data);

	// for(unsigned c = 0; c < num_sequences_; ++ c)
	// {
	// 	for(unsigned n = 0; n < num_dims_; ++ n)
	// 	{
	// 		std::cout << c << " " << n << std::endl;
	// 		std::cout << arrayK_[c][n] << std::endl;
	// 	}
	// 	std::cout << std::endl;
	// }
	// for(unsigned c = 0; c < num_sequences_; ++ c)
	// {
	// 	for(unsigned n = 0; n < num_dims_; ++ n)
	// 	{
	// 		for(unsigned i = 0; i <= all_timestamp_per_dimension_[c][n].size(); ++ i)
	// 		{
	// 			for(std::vector<std::pair<double, unsigned> >::const_iterator p_i = arrayG_[c][n][i].begin(); p_i != arrayG_[c][n][i].end(); ++ p_i)
	// 			{
	// 				std::cout << p_i->first << "," << p_i->second << "; ";
	// 			}
	// 			std::cout << std::endl;
	// 		}
	// 		std::cout << std::endl;
	// 	}
	// 	// std::cout << std::endl;
	// }

	options_ = options;

	Optimizer opt(this);

	opt.PLBFGS(0, 1e10);

	PostProcessing();


}

void SelfInhibitingProcess::PostProcessing()
{
	Eigen::Map<Eigen::MatrixXd> Beta = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	Eigen::MatrixXd Alpha = Beta;
	double epsilon = 5e-2;
	Eigen::VectorXd colsum = Alpha.colwise().sum();
	colsum = (colsum.array() > 0).select(colsum, 1);
	Alpha = Alpha.array().rowwise() / colsum.transpose().array();
	Alpha = (Alpha.array() < epsilon).select(0, Alpha);
	Alpha = (Alpha.array() >= epsilon).select(1, Alpha);

	std::cout << "Estimated Structure : " << std::endl;

	std::cout << Alpha.cast<unsigned>() << std::endl << std::endl;

	Beta = (Alpha.array() == 0).select(0, Beta);

}

double SelfInhibitingProcess::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	return 0;
}

double SelfInhibitingProcess::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	return 0;	
}

void SelfInhibitingProcess::RestoreOptionToDefault()
{
	options_.base_intensity_regularizer = NONE;
	options_.excitation_regularizer = NONE;
	options_.coefficients[LAMBDA0] = 0;
	options_.coefficients[LAMBDA] = 0;
}

