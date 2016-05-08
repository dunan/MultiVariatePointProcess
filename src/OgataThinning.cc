/**
 * \file OgataThinning.cc
 * \brief The class implementation of OgataThinning implementing Ogata's thinning algorithm.
 */
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "../include/OgataThinning.h"
#include "../include/Utility.h"

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
			Eigen::VectorXd intensity_upper_dim;

			const double& lambda_star = process.IntensityUpperBound(t, step_, seq, intensity_upper_dim);
			
			double s = RNG_.GetExponential(1 / lambda_star);

			Eigen::VectorXd intensity_dim;
			
			const double& lambda_t = process.Intensity(t + s, seq, intensity_dim);

			if(s > step_)
			{

				t += step_;

			}
			else if ((t + s > *i_vec_T) || (RNG_.GetUniform() > (lambda_t / lambda_star)))
			{

				t += s;

			}
			else
			{
				std::vector<double> cumprob(num_dims_,0);

				double p = 0;

				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					cumprob[d] = p + intensity_dim(d) / lambda_t;
					p += intensity_dim(d) / lambda_t;
				}

				Event event;
				event.EventID = eventID;
				++ eventID;
				event.SequenceID = sequenceID;

				double D = RNG_.GetUniform();
				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					if(D <= cumprob[d])
					{
						event.DimentionID = d;
						break;
					}
				}

				t += s;

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

		while(eventID <= n)
		{
			Eigen::VectorXd intensity_upper_dim;

			const double& lambda_star = process.IntensityUpperBound(t, step_, seq, intensity_upper_dim);
			
			double s = RNG_.GetExponential(1 / lambda_star);

			Eigen::VectorXd intensity_dim;
			
			const double& lambda_t = process.Intensity(t + s, seq, intensity_dim);

			if(s > step_)
			{

				t += step_;

			}
			else if (RNG_.GetUniform() > (lambda_t / lambda_star))
			{
				t += s;
			}
			else
			{
				std::vector<double> cumprob(num_dims_,0);

				double p = 0;

				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					cumprob[d] = p + intensity_dim(d) / lambda_t;
					p += intensity_dim(d) / lambda_t;
				}

				Event event;
				event.EventID = eventID;
				++ eventID;
				event.SequenceID = sequenceID;

				double D = RNG_.GetUniform();
				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					if(D <= cumprob[d])
					{
						event.DimentionID = d;
						break;
					}
				}

				t += s;

				event.time = t;
				event.marker = 0;

				seq.Add(event);
			}
		}

		seq.PopBack();

		sequences.push_back(seq);

	}
}



Event OgataThinning::SimulateNext(IProcess& process, const Sequence& seq)
{
	const std::vector<Event>& events = seq.GetEvents();

	if(events.size() > 0)
	{
		double t = events[events.size() - 1].time;	

		int eventID = events.size() - 1;

		while(true)
		{
			Eigen::VectorXd intensity_upper_dim;

			const double& lambda_star = process.IntensityUpperBound(t, step_, seq, intensity_upper_dim);
			
			double s = RNG_.GetExponential(1 / lambda_star);

			Eigen::VectorXd intensity_dim;
			
			const double& lambda_t = process.Intensity(t + s, seq, intensity_dim);

			if(s > step_)
			{

				t += step_;

			}
			else if (RNG_.GetUniform() > (lambda_t / lambda_star))
			{

				t += s;

			}
			else
			{
				std::vector<double> cumprob(num_dims_,0);

				double p = 0;

				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					cumprob[d] = p + intensity_dim(d) / lambda_t;
					p += intensity_dim(d) / lambda_t;
				}

				Event event;
				event.EventID = eventID;
				++ eventID;
				event.SequenceID = -1;

				double D = RNG_.GetUniform();
				for(unsigned d = 0; d < num_dims_; ++ d)
				{
					if(D <= cumprob[d])
					{
						event.DimentionID = d;
						break;
					}
				}

				t += s;

				event.time = t;
				event.marker = 0;

				return event;
			}
		}
	}
	
	Eigen::VectorXd intensity_upper_dim;

	const double& lambda_star = process.IntensityUpperBound(0, step_, seq, intensity_upper_dim);

	Event event;
	event.EventID = 0;
	event.SequenceID = -1;
	event.time = RNG_.GetExponential(1 / lambda_star);
	event.marker = 0;

	return event;

	
}
