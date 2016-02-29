#include <vector>
#include <cmath>
#include "../include/PlainHawkes.h"
#include "../include/Sequence.h"

void PlainHawkes::Initialize(const std::vector<Sequence>& data)
{
	num_sequences_ = data.size();
	const unsigned& num_of_dimensions = IProcess::GetNumDims();

	all_exp_kernel_recursive_sum_ = std::vector<std::vector<std::vector<std::vector<double> > > >(
      num_sequences_,
      std::vector<std::vector<std::vector<double> > >(
          num_of_dimensions,
          std::vector<std::vector<double> >(num_of_dimensions, std::vector<double>())));


	all_timestamp_per_dimension_ = std::vector<std::vector<std::vector<double> > >(num_sequences_, std::vector<std::vector<double> > (num_of_dimensions, std::vector<double> ()));

	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();

		for(unsigned i = 0; i < seq.size(); ++ i)
		{
			all_timestamp_per_dimension_[c][seq[i].DimentionID].push_back(seq[i].time);
		}

	}

	for (unsigned k = 0; k < num_sequences_; ++k) {

	    for (unsigned m = 0; m < num_of_dimensions; ++m) {

	      for (unsigned n = 0; n < num_of_dimensions; ++n) {

	        all_exp_kernel_recursive_sum_[k][m][n].push_back(0);

	        if (m != n) {
	          for (unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size();
	               ++i) {
	            double value = exp(-(all_timestamp_per_dimension_[k][n][i] -
	                                 all_timestamp_per_dimension_[k][n][i - 1])) *
	                           all_exp_kernel_recursive_sum_[k][m][n][i - 1];

	            for (unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size();
	                 ++j) {
	              if ((all_timestamp_per_dimension_[k][n][i - 1] <=
	                   all_timestamp_per_dimension_[k][m][j]) &&
	                  (all_timestamp_per_dimension_[k][m][j] <
	                   all_timestamp_per_dimension_[k][n][i])) {
	                value += exp(-(all_timestamp_per_dimension_[k][n][i] -
	                               all_timestamp_per_dimension_[k][m][j]));
	              }
	            }

	            all_exp_kernel_recursive_sum_[k][m][n].push_back(value);
	          }
	        } else {
	          for (unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size();
	               ++i) {
	            double value = exp(-(all_timestamp_per_dimension_[k][n][i] -
	                                 all_timestamp_per_dimension_[k][n][i - 1])) *
	                           (1 + all_exp_kernel_recursive_sum_[k][m][n][i - 1]);
	            all_exp_kernel_recursive_sum_[k][m][n].push_back(value);
	          }
	        }
	      }
	    }
  	}

  	intensity_itegral_features_ = std::vector<std::vector<std::vector<double> > > (num_sequences_, std::vector<std::vector<double> >(num_of_dimensions, std::vector<double>(num_of_dimensions, 0)));

  	for (unsigned c = 0; c < num_sequences_; ++c) {

  		const double& Tc = data[c].GetTimeWindow();

	    for (unsigned m = 0; m < num_of_dimensions; ++m) {

	      for (unsigned n = 0; n < num_of_dimensions; ++n) {

	      	for (unsigned i = 0; i < all_timestamp_per_dimension_[c][n].size(); ++i) {

	      		intensity_itegral_features_[c][m][n] += (1 - exp(-beta_[Idx(m,n)] * (Tc - all_timestamp_per_dimension_[c][n][i])));

	      	}
	      }
	  	}
	}


}

unsigned PlainHawkes::Idx(const unsigned& i, const unsigned& j)
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
				unsigned idx = Idx(d, seq[i].DimentionID);
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
	return 0;
}

void PlainHawkes::NegLoglikelihood(double& objvalue, std::vector<double>& gradient)
{
	
	

}