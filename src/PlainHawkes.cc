#include <vector>
#include <cmath>
#include <iostream>
#include "../include/PlainHawkes.h"
#include "../include/Sequence.h"

void PlainHawkes::Initialize(const std::vector<Sequence>& data)
{
	num_sequences_ = data.size();
	const unsigned& num_of_dimensions = IProcess::GetNumDims();

	all_exp_kernel_recursive_sum_ = std::vector<std::vector<std::vector<std::vector<double> > > >(num_sequences_, std::vector<std::vector<std::vector<double> > >(
          num_of_dimensions, std::vector<std::vector<double> >(num_of_dimensions, std::vector<double>())));


	all_timestamp_per_dimension_ = std::vector<std::vector<std::vector<double> > >(num_sequences_, std::vector<std::vector<double> > (num_of_dimensions, std::vector<double> ()));

	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();

		for(unsigned i = 0; i < seq.size(); ++ i)
		{
			all_timestamp_per_dimension_[c][seq[i].DimentionID].push_back(seq[i].time);
		}

	}

	for (unsigned k = 0; k < num_sequences_; ++k) 
	{

	    for (unsigned m = 0; m < num_of_dimensions; ++m) 
	    {

	      for (unsigned n = 0; n < num_of_dimensions; ++n) 
	      {

	        all_exp_kernel_recursive_sum_[k][m][n].push_back(0);

	        if (m != n) {

	          for (unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size(); ++i) 
	          {

	            double value = exp(-beta_[PlainHawkes::Idx(m, n)] * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][n][i - 1])) * all_exp_kernel_recursive_sum_[k][m][n][i - 1];

	            for (unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size(); ++j) {
	              if ((all_timestamp_per_dimension_[k][n][i - 1] <= all_timestamp_per_dimension_[k][m][j]) &&
	                  (all_timestamp_per_dimension_[k][m][j] < all_timestamp_per_dimension_[k][n][i])) 
	              {
	                value += exp(-beta_[PlainHawkes::Idx(m, n)] * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][m][j]));
	              }
	            }

	            all_exp_kernel_recursive_sum_[k][m][n].push_back(value);
	          }
	        } else 
	        {
	          for (unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size(); ++i) {
	            double value = exp(-beta_[PlainHawkes::Idx(m, n)] * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][n][i - 1])) * (1 + all_exp_kernel_recursive_sum_[k][m][n][i - 1]);
	            all_exp_kernel_recursive_sum_[k][m][n].push_back(value);
	          }
	        }
	      }
	    }
  	}

  	intensity_itegral_features_ = std::vector<std::vector<std::vector<double> > > (num_sequences_, std::vector<std::vector<double> >(num_of_dimensions, std::vector<double>(num_of_dimensions, 0)));

  	observation_window_T_ = std::vector<double> (num_of_dimensions, 0);

  	for (unsigned c = 0; c < num_sequences_; ++c) {

  		observation_window_T_[c] = data[c].GetTimeWindow();

	    for (unsigned m = 0; m < num_of_dimensions; ++ m) {

	      for (unsigned n = 0; n < num_of_dimensions; ++ n) {

	      	for (unsigned i = 0; i < all_timestamp_per_dimension_[c][m].size(); ++i) {

	      		intensity_itegral_features_[c][m][n] += (1 - exp(-beta_[PlainHawkes::Idx(m, n)] * (observation_window_T_[c] - all_timestamp_per_dimension_[c][m][i])));

	      	}
	      }
	  	}
	}
}

unsigned PlainHawkes::PlainHawkes::Idx(const unsigned& i, const unsigned& j)
{
	return (i * IProcess::GetNumDims() + j);
}

double PlainHawkes::Intensity(const double& t, const Sequence& data, std::vector<double>& intensity_dim)
{
	
	const unsigned& num_of_dimensions = IProcess::GetNumDims();

	intensity_dim = std::vector<double>(num_of_dimensions, 0);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix
	const std::vector<double>& params = IProcess::GetParameters();

	for(unsigned d = 0; d < num_of_dimensions; ++ d)
	{
		intensity_dim[d] = params[d];
	}

	const std::vector<Event>& seq = data.GetEvents();

	for(unsigned i = 0; i < seq.size(); ++ i)
	{
		if (seq[i].time < t)
		{
			for(unsigned d = 0; d < num_of_dimensions; ++ d)
			{
				unsigned idx = PlainHawkes::Idx(seq[i].DimentionID, d);
				intensity_dim[d] += params[num_of_dimensions + idx] * exp(-beta_[idx] * (t - seq[i].time));
			}	
		}else
		{
			break;
		}
	}

	double total = 0;
	for(unsigned d = 0; d < num_of_dimensions; ++ d)
	{
		total += intensity_dim[d];
	}

	return total;
	
}

double PlainHawkes::IntensityUpperBound(const double& t, const Sequence& data, std::vector<double>& intensity_upper_dim)
{
	const unsigned& num_of_dimensions = IProcess::GetNumDims();

	intensity_upper_dim = std::vector<double>(num_of_dimensions, 0);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix
	const std::vector<double>& params = IProcess::GetParameters();

	for(unsigned d = 0; d < num_of_dimensions; ++ d)
	{
		intensity_upper_dim[d] = params[d];
	}

	const std::vector<Event>& seq = data.GetEvents();

	for(unsigned i = 0; i < seq.size(); ++ i)
	{
		// pay attention to the event that just happened, so we need consider the = sign
		if (seq[i].time <= t)
		{
			for(unsigned d = 0; d < num_of_dimensions; ++ d)
			{
				unsigned idx = PlainHawkes::Idx(seq[i].DimentionID, d);
				intensity_upper_dim[d] += params[num_of_dimensions + idx] * exp(-beta_[idx] * (t - seq[i].time));
			}	
		}else
		{
			break;
		}
	}

	double total = 0;
	for(unsigned d = 0; d < num_of_dimensions; ++ d)
	{
		total += intensity_upper_dim[d];
	}

	return total;
}

void PlainHawkes::NegLoglikelihood(double& objvalue, std::vector<double>& gradient)
{

	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	const unsigned& num_of_dimensions = IProcess::GetNumDims();

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix
	const std::vector<double>& params = IProcess::GetParameters();

	std::vector<double> grad_lambda0_vector(num_of_dimensions, 0);

	std::vector<double> grad_alpha_matrix(num_of_dimensions * num_of_dimensions, 0);

	std::vector<double> lambda0_vector(params.begin(), params.begin() + num_of_dimensions);

	std::vector<double> alpha_matrix(params.begin() + num_of_dimensions, params.end());

	objvalue = 0;

	for (unsigned k = 0; k < num_sequences_; ++k) 
	{
	    const std::vector<std::vector<double> > &timestamp_per_dimension = all_timestamp_per_dimension_[k];

	    const std::vector<std::vector<std::vector<double> > > &exp_kernel_recursive_sum = all_exp_kernel_recursive_sum_[k];

	    for (unsigned n = 0; n < num_of_dimensions; ++n) 
	    {

	      double obj_n = 0;

	      for (unsigned i = 0; i < timestamp_per_dimension[n].size(); ++i) 
	      {
	        double local_sum = lambda0_vector[n];
	        
	        for (unsigned m = 0; m < num_of_dimensions; ++m) 
	        {
	          local_sum += alpha_matrix[PlainHawkes::Idx(m, n)] * exp_kernel_recursive_sum[m][n][i];
	        }

	        obj_n += log(local_sum);

	        grad_lambda0_vector[n] += (1 / local_sum);

	        for (unsigned m = 0; m < num_of_dimensions; ++m) 
	        {
	          grad_alpha_matrix[PlainHawkes::Idx(m, n)] += exp_kernel_recursive_sum[m][n][i] / local_sum;
	        }
	      }

	      for (unsigned m = 0; m < num_of_dimensions; ++ m) 
	      {
	      	const unsigned& idx = PlainHawkes::Idx(m, n);

	        obj_n -= (alpha_matrix[idx] / beta_[idx]) * intensity_itegral_features_[k][m][n];

	        grad_alpha_matrix[idx] -= intensity_itegral_features_[k][m][n] / beta_[idx];
	      }

	      obj_n -= observation_window_T_[k] * lambda0_vector[n];

	      grad_lambda0_vector[n] -= observation_window_T_[k];

	      objvalue += obj_n;
	    }
  	}

  	gradient = std::vector<double>(grad_lambda0_vector.begin(), grad_lambda0_vector.end());

  	gradient.insert(gradient.end(), grad_alpha_matrix.begin(), grad_alpha_matrix.end());

  	for(unsigned i = 0; i < gradient.size(); ++ i)
  	{
  		gradient[i] = (-gradient[i]) / num_sequences_;
  	}

	objvalue = -objvalue / num_sequences_;

}

void PlainHawkes::Gradient(const unsigned &k, std::vector<double>& gradient)
{
	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	const unsigned& num_of_dimensions = IProcess::GetNumDims();

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix
	const std::vector<double>& params = IProcess::GetParameters();

	std::vector<double> grad_lambda0_vector(num_of_dimensions, 0);

	std::vector<double> grad_alpha_matrix(num_of_dimensions * num_of_dimensions, 0);

	std::vector<double> lambda0_vector(params.begin(), params.begin() + num_of_dimensions);

	std::vector<double> alpha_matrix(params.begin() + num_of_dimensions, params.end());

	const std::vector<std::vector<double> > &timestamp_per_dimension = all_timestamp_per_dimension_[k];

    const std::vector<std::vector<std::vector<double> > > &exp_kernel_recursive_sum = all_exp_kernel_recursive_sum_[k];

	for (unsigned n = 0; n < num_of_dimensions; ++n) 
    {
      for (unsigned i = 0; i < timestamp_per_dimension[n].size(); ++i) 
      {
        double local_sum = lambda0_vector[n];
        
        for (unsigned m = 0; m < num_of_dimensions; ++m) 
        {
          local_sum += alpha_matrix[PlainHawkes::Idx(m, n)] * exp_kernel_recursive_sum[m][n][i];
        }

        grad_lambda0_vector[n] += (1 / local_sum);

        for (unsigned m = 0; m < num_of_dimensions; ++m) 
        {
          grad_alpha_matrix[PlainHawkes::Idx(m, n)] += exp_kernel_recursive_sum[m][n][i] / local_sum;
        }
      }

      for (unsigned m = 0; m < num_of_dimensions; ++ m) 
      {
      	const unsigned& idx = PlainHawkes::Idx(m, n);

        grad_alpha_matrix[idx] -= intensity_itegral_features_[k][m][n] / beta_[idx];
      }

      grad_lambda0_vector[n] -= observation_window_T_[k];

    }    

    gradient = std::vector<double>(grad_lambda0_vector.begin(), grad_lambda0_vector.end());

  	gradient.insert(gradient.end(), grad_alpha_matrix.begin(), grad_alpha_matrix.end());

  	for(unsigned i = 0; i < gradient.size(); ++ i)
  	{
  		gradient[i] = (-gradient[i]) / num_sequences_;
  	}
}

