#include <cmath>
#include "../include/Optimizer.h"

void Optimizer::SGD(IProcess* process, const double& gamma0, const unsigned& ini_max_iter, const std::vector<Sequence>& data)
{
	const unsigned& num_of_dimensions = process->GetNumDims();

	std::vector<double> returned_params((process->GetParameters()).begin(), (process->GetParameters()).end());

	unsigned num_sequences = data.size();

	for (std::vector<double>::iterator value = returned_params.begin(); value != returned_params.end(); ++value) {
    	*value = RNG_.GetUniform();
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
		std::vector<double> local_p0 = returned_params;

		std::vector<double> last_local_p0 = returned_params;

		process->SetParameters(local_p0);

		double old_diff = 0;

    	double new_diff = 0;

    	for (unsigned iter = 0; iter < ini_max_iter; ++iter) 
    	{
    		for (unsigned i = 0; i < num_sequences; ++i) 
    		{
    			// gamma = sqrt(ini_gamma / (ini_gamma + t + 1));

    			gamma = ini_gamma;

    			process->SetParameters(local_p0);

    			std::vector<double> grad;
    			process->Gradient(i, grad);

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
    			
    			process->NegLoglikelihood(objvalue, temp);

    			std::cout << "finish epoch : " << iter << " " << objvalue << std::endl;	
    			random_shuffle(idx.begin(), idx.end());
    		}
    	}

    	if (stop) 
    	{
			process->SetParameters(local_p0);
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