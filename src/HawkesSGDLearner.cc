#include <iostream>
#include <cmath>
#include "../include/HawkesSGDLearner.h"

void HawkesSGDLearner::StochasticGradient(const unsigned &k, std::vector<double>& gradient)
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

void HawkesSGDLearner::fit(const std::vector<Sequence>& data, std::vector<double>& returned_params)
{
	PlainHawkes::Initialize(data);

	const unsigned& num_of_dimensions = IProcess::GetNumDims();

	returned_params = std::vector<double>(num_of_dimensions * (1 + num_of_dimensions), 0);

	for (std::vector<double>::iterator value = returned_params.begin(); value != returned_params.end(); ++value) {
    	*value = RNG_.GetUniform();
	}

	// projected SGD
  	std::vector<unsigned> idx(num_sequences_, 0);
  	for (unsigned i = 0; i < num_sequences_; ++i) {
    	idx[i] = i;
  	}

	unsigned t = 0;

	bool stop = true;

	double gamma = 0;

	ini_gamma_ = 1e-5;

	while(true)
	{
		std::vector<double> local_p0 = returned_params;

		std::vector<double> last_local_p0 = returned_params;

		IProcess::SetParameters(local_p0);

		double old_diff = 0;

    	double new_diff = 0;

    	for (unsigned iter = 0; iter < ini_max_iter_; ++iter) 
    	{
    		for (unsigned i = 0; i < num_sequences_; ++i) 
    		{
    			// gamma = sqrt(ini_gamma_ / (ini_gamma_ + t + 1));

    			gamma = ini_gamma_;

    			IProcess::SetParameters(local_p0);

    			std::vector<double> grad;
    			StochasticGradient(i, grad);

    			// update and projection
		        for (unsigned k = 0; k < local_p0.size(); ++k) 
		        {
		          local_p0[k] -= grad[k] * gamma;

		          if (local_p0[k] < 1e-16) 
		          {
		            local_p0[k] = 1e-16;
		          }
		        }

		        std::vector<double> diff(num_of_dimensions * (1 + num_of_dimensions), 0);
		        for (unsigned k = 0; k < diff.size(); ++k) 
		        {
          			diff[k] = local_p0[k] - last_local_p0[k];
        		}

        		if (t == 0) 
        		{
		          old_diff = 0;

		          for (std::vector<double>::const_iterator v = diff.begin(); v != diff.end(); ++v) 
		          {
		            old_diff += (*v) * (*v);
		          }

		          old_diff = sqrt(old_diff);

		        } 
		        else 
		        {
		          new_diff = 0;

		          for (std::vector<double>::const_iterator v = diff.begin(); v != diff.end(); ++v) 
		          {
		            new_diff += (*v) * (*v);
		          }

		          new_diff = sqrt(new_diff);

		          if ((new_diff - old_diff) / old_diff > 1e8) {
		            stop = false;
		            std::cout << "rerun" << std::endl;
		            break;
		          }

		          old_diff = new_diff;
		        }

		        last_local_p0 = local_p0;

        		++t;
    		}

    		if(!stop)
    		{
    			break;
    		}else
    		{
    			double objvalue;
    			std::vector<double> temp;
    			
    			PlainHawkes::NegLoglikelihood(objvalue, temp);

    			std::cout << "finish epoch : " << iter << " " << objvalue << std::endl;	
    			random_shuffle(idx.begin(), idx.end());
    		}
    	}

    	if (stop) 
    	{
			returned_params = local_p0;
			break;

		} else 
		{
			ini_gamma_ /= 2;
			stop = true;
			t = 0;
			gamma = 0;
		}
	}
}