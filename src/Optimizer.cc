#include <cmath>
#include "../include/Optimizer.h"

void Optimizer::SGD(const double& gamma0, const unsigned& ini_max_iter, const std::vector<Sequence>& data)
{
	const unsigned& num_of_dimensions = process_->GetNumDims();

	Eigen::VectorXd returned_params(process_->GetParameters());

	unsigned num_sequences = data.size();

	for (unsigned i = 0; i < returned_params.size(); ++ i) 
	{
		returned_params(i) = RNG_.GetUniform();
	}

	// projected SGD
  	std::vector<unsigned> idx(num_sequences, 0);
  	for (unsigned i = 0; i < num_sequences; ++i) {
    	idx[i] = i;
  	}

	unsigned t = 0;

	bool stop = true;

	double gamma = 0;

	double ini_gamma = gamma0;

	while(true)
	{
		Eigen::VectorXd local_p0 = returned_params;

		Eigen::VectorXd last_local_p0 = returned_params;

		process_->SetParameters(local_p0);

		double old_diff = 0;

    	double new_diff = 0;

    	for (unsigned iter = 0; iter < ini_max_iter; ++iter) 
    	{
    		for (unsigned i = 0; i < num_sequences; ++i) 
    		{
    			// gamma = sqrt(ini_gamma / (ini_gamma + t + 1));

    			gamma = ini_gamma;

    			process_->SetParameters(local_p0);

    			Eigen::VectorXd grad;
    			process_->Gradient(i, grad);

    			// update and projection
    			local_p0 = local_p0.array() - grad.array() * gamma;
    			local_p0 = (local_p0.array() > 1e-16).select(local_p0, 1e-16);

        		if (t == 0) 
        		{

		          old_diff = (local_p0 - last_local_p0).norm();

		        } 
		        else 
		        {
		          new_diff = (local_p0 - last_local_p0).norm();

		          if ((new_diff - old_diff) / old_diff > 1e8) {
		            stop = false;
		            std::cout << "rerun " << new_diff << " " << old_diff << std::endl;
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
    			Eigen::VectorXd temp;
    			
    			process_->NegLoglikelihood(objvalue, temp);

    			std::cout << "finish epoch : " << iter << " " << objvalue << std::endl;	
    			random_shuffle(idx.begin(), idx.end());
    		}
    	}

    	if (stop) 
    	{
			process_->SetParameters(local_p0);
			break;

		} else 
		{
			ini_gamma /= 2;
			stop = true;
			t = 0;
			gamma = 0;
		}
	}

}

