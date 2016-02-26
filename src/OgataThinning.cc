#include <vector>
#include <iostream>
#include "../include/OgataThinning.h"

void OgataThinning::Simulate(IProcess& process, const std::vector<double>& vec_T, std::vector<Sequence>& sequences)
{

	sequences = std::vector<Sequence>();

	unsigned sequenceID = 0;

	for(std::vector<double>::const_iterator i_vec_T = vec_T.begin(); i_vec_T != vec_T.end(); ++ i_vec_T)
	{
		Sequence seq(*i_vec_T);

		double t = 0;

		unsigned eventID = 0;

		while(t < *i_vec_T)
		{
			std::vector<double> intensity_upper_dim;

			const double& lambda_star = process.IntensityUpperBound(t, seq, intensity_upper_dim);
			
			t += RNG_.GetExponential(1 / lambda_star);

			std::vector<double> intensity_dim;
			
			const double& lambda_t = process.Intensity(t, seq, intensity_dim);

			if(RNG_.GetUniform() <= (lambda_t / lambda_star) && (t < *i_vec_T))
			{
				std::vector<double> cumprob(num_dims_,0);

				double p = 0;

				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					cumprob[d] = p + intensity_dim[d] / lambda_t;
					p += intensity_dim[d] / lambda_t;
				}

				Event event;
				event.EventID = eventID;
				++ eventID;
				event.SequenceID = sequenceID;

				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					if(RNG_.GetUniform() <= cumprob[d])
					{
						event.DimentionID = d;
						break;
					}
				}

				event.time = t;
				event.marker = 0;

				seq.Add(event);
			}	
		}

		sequences.push_back(seq);

		++ sequenceID;		

	}
}

void OgataThinning::Simulate(IProcess& process, const unsigned& n, const unsigned& num_sequences, std::vector<Sequence>& sequences)
{
	sequences = std::vector<Sequence>();

	for(unsigned sequenceID = 0; sequenceID < num_sequences; ++ sequenceID)
	{
		Sequence seq;

		double t = 0;

		unsigned eventID = 0;

		while(eventID < n)
		{
			std::vector<double> intensity_upper_dim;

			const double& lambda_star = process.IntensityUpperBound(t, seq, intensity_upper_dim);
			
			t += RNG_.GetExponential(1 / lambda_star);

			std::vector<double> intensity_dim;
			
			const double& lambda_t = process.Intensity(t, seq, intensity_dim);

			if(RNG_.GetUniform() <= (lambda_t / lambda_star))
			{
				std::vector<double> cumprob(num_dims_,0);

				double p = 0;

				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					cumprob[d] = p + intensity_dim[d] / lambda_t;
					p += intensity_dim[d] / lambda_t;
				}

				Event event;
				event.EventID = eventID;
				++ eventID;
				event.SequenceID = sequenceID;

				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					if(RNG_.GetUniform() <= cumprob[d])
					{
						event.DimentionID = d;
						break;
					}
				}

				event.time = t;
				event.marker = 0;

				seq.Add(event);
			}	
		}

		sequences.push_back(seq);	

	}
}

