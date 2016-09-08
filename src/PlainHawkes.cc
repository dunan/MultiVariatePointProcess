/**
 * \file PlainHawkes.cc
 * \brief The class implementation of PlainHawkes implementing the standard Hawkes process.
 */
#include <vector>
#include <cmath>
#include <iostream>

#include <cassert>

#include "../include/PlainHawkes.h"
#include "../include/Sequence.h"
#include "../include/Optimizer.h"
#include "../include/OgataThinning.h"
#include "../include/Utility.h"
#include "../include/GNUPlotWrapper.h"
#include "../include/SimpleRNG.h"

void PlainHawkes::Initialize(const std::vector<Sequence>& data)
{
	num_sequences_ = data.size();

	all_exp_kernel_recursive_sum_ = std::vector<std::vector<std::vector<Eigen::VectorXd> > >(num_sequences_, std::vector<std::vector<Eigen::VectorXd> >(
          num_dims_, std::vector<Eigen::VectorXd>(num_dims_, Eigen::VectorXd())));

	// all_timestamp_per_dimension_ = std::vector<std::vector<std::vector<double> > >(num_sequences_, std::vector<std::vector<double> > (num_dims_, std::vector<double> ()));
	// for(unsigned c = 0; c < num_sequences_; ++ c)
	// {
	// 	const std::vector<Event>& seq = data[c].GetEvents();

	// 	for(unsigned i = 0; i < seq.size(); ++ i)
	// 	{
	// 		all_timestamp_per_dimension_[c][seq[i].DimentionID].push_back(seq[i].time);
	// 	}

	// }

	InitializeDimension(data);

	for (unsigned k = 0; k < num_sequences_; ++k) 
	{
		for (unsigned m = 0; m < num_dims_; ++m) 
		{
			for (unsigned n = 0; n < num_dims_; ++n) 
			{

				if (all_timestamp_per_dimension_[k][n].size() > 0)
				{
					all_exp_kernel_recursive_sum_[k][m][n] = Eigen::VectorXd::Zero(all_timestamp_per_dimension_[k][n].size());

					if (m != n) 
					{
						// handle events on other dimensions that occur before the first event of dimension n
						for (unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size(); ++j) 
						{
							if (all_timestamp_per_dimension_[k][m][j] < all_timestamp_per_dimension_[k][n][0])
							{
								all_exp_kernel_recursive_sum_[k][m][n](0) += exp(-Beta_(m,n) * (all_timestamp_per_dimension_[k][n][0] - all_timestamp_per_dimension_[k][m][j]));
							}
						}

						for (unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size(); ++i) 
						{

							double value = exp(-Beta_(m,n) * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][n][i - 1])) * all_exp_kernel_recursive_sum_[k][m][n](i - 1);

							for (unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size(); ++j) 
							{
								if ((all_timestamp_per_dimension_[k][n][i - 1] <= all_timestamp_per_dimension_[k][m][j]) &&
								  (all_timestamp_per_dimension_[k][m][j] < all_timestamp_per_dimension_[k][n][i])) 
								{
									value += exp(-Beta_(m,n) * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][m][j]));
								}
							}

							all_exp_kernel_recursive_sum_[k][m][n](i) = value;
						}

					} else 
					{
						for (unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size(); ++i) 
						{
							all_exp_kernel_recursive_sum_[k][m][n](i) = exp(-Beta_(m,n) * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][n][i - 1])) * (1 + all_exp_kernel_recursive_sum_[k][m][n](i - 1));
						}
					}
				}
			}
		}
	}

	observation_window_T_ = Eigen::VectorXd::Zero(num_sequences_);

  	intensity_itegral_features_ = std::vector<Eigen::MatrixXd> (num_sequences_, Eigen::MatrixXd::Zero(num_dims_, num_dims_));

  	for (unsigned c = 0; c < num_sequences_; ++c) {

  		observation_window_T_(c) = data[c].GetTimeWindow();

	    for (unsigned m = 0; m < num_dims_; ++ m) {

	      for (unsigned n = 0; n < num_dims_; ++ n) {

	      	Eigen::Map<Eigen::VectorXd> event_dim_m = Eigen::Map<Eigen::VectorXd>(all_timestamp_per_dimension_[c][m].data(), all_timestamp_per_dimension_[c][m].size());

	      	intensity_itegral_features_[c](m,n) = (1 - (-Beta_(m,n) * (observation_window_T_(c) - event_dim_m.array())).exp()).sum();

	      }
	  	}
	}

}


double PlainHawkes::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
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
				intensity_dim(d) += Alpha_(seq[i].DimentionID, d) * exp(-Beta_(seq[i].DimentionID, d) * (t - seq[i].time));
			}	
		}
		else
		{
			break;
		}
	}

	return intensity_dim.array().sum();
	
}

double PlainHawkes::IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
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
				intensity_upper_dim(d) += Alpha_(seq[i].DimentionID, d) * exp(-Beta_(seq[i].DimentionID, d) * (t - seq[i].time));
			}	
		}
		else
		{
			break;
		}
	}

	return intensity_upper_dim.array().sum();
}

void PlainHawkes::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
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

	for (unsigned k = 0; k < num_sequences_; ++k) 
	{
	    const std::vector<std::vector<double> > &timestamp_per_dimension = all_timestamp_per_dimension_[k];

	    const std::vector<std::vector<Eigen::VectorXd> > &exp_kernel_recursive_sum = all_exp_kernel_recursive_sum_[k];

	    for (unsigned n = 0; n < num_dims_; ++n) 
	    {

	      double obj_n = 0;

	      for (unsigned i = 0; i < timestamp_per_dimension[n].size(); ++i) 
	      {
	        double local_sum = Lambda0_(n) + 1e-4;
	        
	        for (unsigned m = 0; m < num_dims_; ++m) 
	        {
	          local_sum += Alpha_(m,n) * exp_kernel_recursive_sum[m][n](i);
	        }

	        obj_n += log(local_sum);

	        grad_lambda0_vector(n) += (1 / local_sum);

	        for (unsigned m = 0; m < num_dims_; ++m) 
	        {
	          grad_alpha_matrix(m, n) += exp_kernel_recursive_sum[m][n](i) / local_sum;
	        }
	      }

	      obj_n -= ((Alpha_.col(n).array() / Beta_.col(n).array()) * intensity_itegral_features_[k].col(n).array()).sum();

	      grad_alpha_matrix.col(n) = grad_alpha_matrix.col(n).array() - (intensity_itegral_features_[k].col(n).array() / Beta_.col(n).array());

	      obj_n -= observation_window_T_(k) * Lambda0_(n);

	      grad_lambda0_vector(n) -= observation_window_T_(k);

	      objvalue += obj_n;

	    }
  	}

  	gradient = -gradient.array() / num_sequences_;

	objvalue = -objvalue / num_sequences_;

	// Regularization for base intensity
	switch (options_.base_intensity_regularizer)
	{
		case L22 :
			
			grad_lambda0_vector = grad_lambda0_vector.array() + (options_.coefficients[LAMBDA] * Lambda0_.array());

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA] * Lambda0_.squaredNorm();
			
			break;

		case L1 :

			grad_lambda0_vector = grad_lambda0_vector.array() + options_.coefficients[LAMBDA];

			objvalue = objvalue + options_.coefficients[LAMBDA] * Lambda0_.array().abs().sum();

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

			grad_alpha_vector = grad_alpha_vector.array() + (options_.coefficients[BETA] * alpha_vector.array());

			objvalue = objvalue + 0.5 * options_.coefficients[BETA] * alpha_vector.squaredNorm();

			break;

		case L1 :

			grad_alpha_vector = grad_alpha_vector.array() + options_.coefficients[BETA];

			objvalue = objvalue + options_.coefficients[BETA] * alpha_vector.array().abs().sum();

			break;

		default:
			break;
	}

}

void PlainHawkes::Gradient(const unsigned &k, Eigen::VectorXd& gradient)
{
	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	// // first num_of_dimensions of the parameters are the base intensity
	// // the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	gradient = Eigen::VectorXd::Zero(num_dims_ * (1 + num_dims_));

	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> grad_alpha_matrix = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	const std::vector<std::vector<double> > &timestamp_per_dimension = all_timestamp_per_dimension_[k];

	const std::vector<std::vector<Eigen::VectorXd> > &exp_kernel_recursive_sum = all_exp_kernel_recursive_sum_[k];

	for (unsigned n = 0; n < num_dims_; ++n) 
    {
      for (unsigned i = 0; i < timestamp_per_dimension[n].size(); ++i) 
      {
        double local_sum = Lambda0_(n);
        
        for (unsigned m = 0; m < num_dims_; ++m) 
        {
          local_sum += Alpha_(m,n) * exp_kernel_recursive_sum[m][n](i);
        }

        grad_lambda0_vector(n) += (1 / local_sum);

        for (unsigned m = 0; m < num_dims_; ++m) 
        {
          grad_alpha_matrix(m, n) += exp_kernel_recursive_sum[m][n](i) / local_sum;
        }
      }

      grad_alpha_matrix.col(n) = grad_alpha_matrix.col(n).array() - (intensity_itegral_features_[k].col(n).array() / Beta_.col(n).array());

      grad_lambda0_vector(n) -= observation_window_T_(k);

    }    

    gradient = -gradient.array() / num_sequences_;
}

void PlainHawkes::fit(const std::vector<Sequence>& data, const OPTION& options)
{
	PlainHawkes::Initialize(data);

	options_ = options;

	Optimizer opt(this);

	switch (options_.excitation_regularizer)
	{
		case NUCLEAR :
		{
			opt.ProximalFrankWolfe(options_.ini_learning_rate, options_.coefficients[BETA], options_.rho, options_.ub_nuclear, options_.ini_max_iter);
			break;
		}

		default :	
		{
			assert(options_.base_intensity_regularizer != NUCLEAR);

			switch (options.method)
			{
				case SGD:
					opt.SGD(options_.ini_learning_rate, options_.ini_max_iter, data);
					break;

				case PLBFGS:
					opt.PLBFGS(0, 1e10);
					break;
			}

			break;
		}

	}

	RestoreOptionToDefault();

}

void PlainHawkes::fit(const std::vector<Sequence>& data, const OPTION& options, const Eigen::VectorXd& trueparameters)
{
	assert(options.base_intensity_regularizer != NUCLEAR);

	PlainHawkes::Initialize(data);

	options_ = options;

	Optimizer opt(this);

	switch (options_.excitation_regularizer)
	{
		case NUCLEAR :
		{
			opt.ProximalFrankWolfe(options_.ini_learning_rate, options_.coefficients[BETA], options_.rho, options_.ub_nuclear, options_.ini_max_iter, trueparameters);
			break;
		}

		default :	
		{
			switch (options.method)
			{
				case SGD:
					opt.SGD(options_.ini_learning_rate, options_.ini_max_iter, data);
					break;

				case PLBFGS:
					opt.PLBFGS(0, 1e10);
					break;
			}

			break;
		}

	}
	
	RestoreOptionToDefault();

}

double PlainHawkes::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
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

double PlainHawkes::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	std::vector<Sequence> sequences;
	sequences.push_back(data);

	InitializeDimension(sequences);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	std::vector<std::vector<double> >& timestamp_per_dimension = all_timestamp_per_dimension_[0];

	double integral_value = 0;

	for(unsigned n = 0; n < num_dims_; ++ n)
	{
		integral_value += Lambda0_(n) * (upper - lower);

		for(unsigned m = 0; m < num_dims_; ++ m)
		{
			Eigen::Map<Eigen::VectorXd> event_dim_m = Eigen::Map<Eigen::VectorXd>(timestamp_per_dimension[m].data(), timestamp_per_dimension[m].size());

			Eigen::VectorXd mask = (event_dim_m.array() < lower).cast<double>();
			double a = (mask.array() * (((-Beta_(m,n) * (lower - event_dim_m.array())) * mask.array()).exp() - ((-Beta_(m,n) * (upper - event_dim_m.array())) * mask.array()).exp())).sum();

			mask = (event_dim_m.array() >= lower && event_dim_m.array() < upper).cast<double>();
			double b = (mask.array() * (1 - ((-Beta_(m,n) * (upper - event_dim_m.array())) * mask.array()).exp())).sum();

			integral_value += (Alpha_(m,n) / Beta_(m,n)) * (a + b);

		}
	}


	return integral_value;
}

void PlainHawkes::RestoreOptionToDefault()
{
	options_.method = PLBFGS;
	options_.base_intensity_regularizer = NONE;
	options_.excitation_regularizer = NONE;
	options_.coefficients[LAMBDA] = 0;
	options_.coefficients[BETA] = 0;
}

unsigned PlainHawkes::AssignDim(const Eigen::VectorXd& intensity_dim)
{
	double total_intensity = intensity_dim.array().sum();

	std::vector<double> cumprob(num_dims_,0);

	double p = 0;

	for(unsigned d = 0; d < num_dims_; ++ d)
	{
		cumprob[d] = p + intensity_dim(d) / total_intensity;
		p += intensity_dim(d) / total_intensity;
	}

	double D = RNG_.GetUniform();
	for(unsigned d = 0; d < num_dims_; ++ d)
	{
		if(D <= cumprob[d])
		{
			return d;
		}
	}
	return num_dims_ - 1;
}

void PlainHawkes::UpdateExpSum(double t, const Eigen::VectorXd& last_event_per_dim, Eigen::MatrixXd& expsum)
{
	// update the intensity function of each dimension since t has been changed
	for(unsigned d = 0; d < num_dims_; ++ d)
	{
		if((last_event_per_dim(d) > 0))
		{
			expsum.row(d) = (-(t - last_event_per_dim(d)) * Beta_.row(d).array()).exp() * (1 + expsum.row(d).array());
		}
	}
}

void PlainHawkes::Simulate(const std::vector<double>& vec_T, std::vector<Sequence>& sequences)
{
	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	sequences = std::vector<Sequence>();

	unsigned sequenceID = 0;

	for(std::vector<double>::const_iterator i_vec_T = vec_T.begin(); i_vec_T != vec_T.end(); ++ i_vec_T)
	{
		Eigen::MatrixXd expsum = Eigen::MatrixXd::Zero(num_dims_, num_dims_);
		Eigen::VectorXd last_event_per_dim = Eigen::VectorXd::Zero(num_dims_);

		Sequence seq(*i_vec_T);

		unsigned eventID = 0;

		double t = RNG_.GetExponential(1.0 / Lambda0_.array().sum());

		if(t < *i_vec_T)
		{
			Event event;
			event.EventID = eventID;
			event.SequenceID = sequenceID;
			event.DimentionID = AssignDim(Lambda0_);
			event.time = t;
			event.marker = 0;
			seq.Add(event);

			last_event_per_dim(event.DimentionID) = t;			

			++ eventID;

			unsigned last_event_dim = event.DimentionID;

			while(t < *i_vec_T)
			{
				Eigen::MatrixXd temp = expsum;

				double lambda_star = Lambda0_.array().sum();

				// update the intensity function of each dimension since t has been changed
				UpdateExpSum(t, last_event_per_dim, temp);

				lambda_star += (Alpha_.array() * temp.array()).sum();
				
				while(true)
				{
					
					t += RNG_.GetExponential(1.0 / lambda_star);

					Eigen::MatrixXd temp = expsum;

					// update the intensity function of each dimension at the new time t
					UpdateExpSum(t, last_event_per_dim, temp);

					Eigen::VectorXd intensity_dim = Lambda0_.array() + (Alpha_.array() * temp.array()).colwise().sum().transpose();

					double D = RNG_.GetUniform();

					if(D <= intensity_dim.array().sum() / lambda_star)
					{
						Event event;
						event.EventID = eventID;
						event.SequenceID = sequenceID;
						event.DimentionID = AssignDim(intensity_dim);
						last_event_dim = event.DimentionID;

						if(last_event_per_dim(last_event_dim) > 0)
						{
							expsum.row(last_event_dim) = (-(t - last_event_per_dim(last_event_dim)) * Beta_.row(last_event_dim).array()).exp() * (1 + expsum.row(last_event_dim).array());	
						}
						
						event.time = t;
						last_event_per_dim(event.DimentionID) = t;	

						event.marker = 0;
						seq.Add(event);
						
						++ eventID;

						break;

					}else
					{
						lambda_star = intensity_dim.array().sum();
					}

				}
				
			}

			sequences.push_back(seq);

			++ sequenceID; 
		}
	}

}

void PlainHawkes::Simulate(const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences)
{
	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	sequences = std::vector<Sequence>();

	for(unsigned sequenceID = 0; sequenceID < num_sequences; ++ sequenceID)
	{
	        Eigen::MatrixXd expsum = Eigen::MatrixXd::Zero(num_dims_, num_dims_);
	        Eigen::VectorXd last_event_per_dim = Eigen::VectorXd::Zero(num_dims_);

		Sequence seq;

		unsigned eventID = 0;

		double t = RNG_.GetExponential(1.0 / Lambda0_.array().sum());

		Event event;
		event.EventID = eventID;
		event.SequenceID = sequenceID;
		event.DimentionID = AssignDim(Lambda0_);
		event.time = t;
		event.marker = 0;
		seq.Add(event);

		last_event_per_dim(event.DimentionID) = t;			

		++ eventID;

		unsigned last_event_dim = event.DimentionID;

		while(eventID <= n)
		{
			Eigen::MatrixXd temp = expsum;

			double lambda_star = Lambda0_.array().sum();

			UpdateExpSum(t, last_event_per_dim, temp);

			lambda_star += (Alpha_.array() * temp.array()).sum();

			while(true)
			{
				t += RNG_.GetExponential(1.0 / lambda_star);

				Eigen::MatrixXd temp = expsum;

				UpdateExpSum(t, last_event_per_dim, temp);

				Eigen::VectorXd intensity_dim = Lambda0_.array() + (Alpha_.array() * temp.array()).colwise().sum().transpose();

				double D = RNG_.GetUniform();

				if(D <= intensity_dim.array().sum() / lambda_star)
				{
					Event event;
					event.EventID = eventID;
					event.SequenceID = sequenceID;
					event.DimentionID = AssignDim(intensity_dim);
					last_event_dim = event.DimentionID;

					if(last_event_per_dim(last_event_dim) > 0)
					{
						expsum.row(last_event_dim) = (-(t - last_event_per_dim(last_event_dim)) * Beta_.row(last_event_dim).array()).exp() * (1 + expsum.row(last_event_dim).array());	
					}
					
					event.time = t;
					last_event_per_dim(event.DimentionID) = t;	

					event.marker = 0;
					seq.Add(event);
					
					++ eventID;

					break;

				}else
				{
					lambda_star = intensity_dim.array().sum();
				}
			}
		}

		seq.PopBack();

		sequences.push_back(seq);

	}

}
