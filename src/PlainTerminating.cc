/**
 * \file PlainTerminating.cc
 * \brief The class implementation of PlainTerminating implementing the Terminating process.
 */
#include <vector>
#include <cmath>
#include <iostream>
#include "../include/PlainTerminating.h"
#include "../include/Sequence.h"
#include "../include/Optimizer.h"
#include "../include/OgataThinning.h"

void PlainTerminating::Initialize(const std::vector<Sequence>& data)
{

	InitializeDimension(data);

	num_sequences_ = data.size();

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		Eigen::MatrixXd matrixK = Eigen::MatrixXd::Zero(num_sequences_, num_dims_);
		Eigen::MatrixXd matrixG = Eigen::MatrixXd::Zero(num_sequences_, num_dims_);

		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			const std::vector<Event>& seq = data[c].GetEvents();
			
			if(all_timestamp_per_dimension_[c][i].size() > 0) // if infected
			{

				const double& i_time = all_timestamp_per_dimension_[c][i][0];

				for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
				{
					const int& j = i_event->DimentionID;
					const double& j_time = i_event->time;

					if(j_time < i_time)
					{
						matrixG(c, j) = i_time - j_time;
						matrixK(c, j) = 1;
					}else
					{
						break;
					}
				}

			}else // if survival
			{
				for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
				{
					const int& j = i_event->DimentionID;
					const double& j_time = i_event->time;

					matrixG(c, j) = data[c].GetTimeWindow() - j_time;
					matrixK(c, j) = 1;
				}
			}	
		}

		arrayK.push_back(matrixK);
		arrayG.push_back(matrixG);
	}
}

void PlainTerminating::InitializeWithGraph(const std::vector<Sequence>& data)
{

	InitializeDimension(data);

	num_sequences_ = data.size();

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		Eigen::MatrixXd matrixK = Eigen::MatrixXd::Zero(num_sequences_, num_dims_);
		Eigen::MatrixXd matrixG = Eigen::MatrixXd::Zero(num_sequences_, num_dims_);

		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			
			if(all_timestamp_per_dimension_[c][i].size() > 0) // if infected
			{

				const double& i_time = all_timestamp_per_dimension_[c][i][0];

				for(std::set<unsigned>::const_iterator i_parent = graph_->nodes[i].parents.begin(); i_parent != graph_->nodes[i].parents.end(); ++ i_parent)
				{
					if((all_timestamp_per_dimension_[c][*i_parent].size() > 0) && (all_timestamp_per_dimension_[c][*i_parent][0] < i_time))
					{
						matrixG(c, *i_parent) = i_time - all_timestamp_per_dimension_[c][*i_parent][0];
						matrixK(c, *i_parent) = 1;
					}
				}

			}else // if survival
			{
				for(std::set<unsigned>::const_iterator i_parent = graph_->nodes[i].parents.begin(); i_parent != graph_->nodes[i].parents.end(); ++ i_parent)
				{
					if(all_timestamp_per_dimension_[c][*i_parent].size() > 0)
					{
						matrixG(c, *i_parent) = data[c].GetTimeWindow() - all_timestamp_per_dimension_[c][*i_parent][0];
						matrixK(c, *i_parent) = 1;
					}
				}
			}	
		}

		arrayK.push_back(matrixK);
		arrayG.push_back(matrixG);
	}
}

void PlainTerminating::PostProcessing()
{
	Eigen::Map<Eigen::MatrixXd> alpha_matrix = Eigen::Map<Eigen::MatrixXd>(parameters_.data(), num_dims_, num_dims_);

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		alpha_matrix(i,i) = 0;
	}

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		Eigen::VectorXd parents = arrayK[i].colwise().sum();

		for(unsigned j = 0; j < num_dims_; ++ j)
		{
			if(parents(j) == 0)
			{
				alpha_matrix(j,i) = 0;	
			}
			
		}
	}

}

//  MLE esitmation of the parameters
void PlainTerminating::fit(const std::vector<Sequence>& data, const OPTION& options)
{

	if(graph_ == NULL)
	{
		PlainTerminating::Initialize(data);	
	}else
	{
		PlainTerminating::InitializeWithGraph(data);
	}

	options_ = options;

	Optimizer opt(this);

	opt.PLBFGS(0, 1e10);

	PlainTerminating::PostProcessing();

	return;
}

//  This virtual function requires process-specific implementation. It calculates the negative loglikelihood of the given data. This function must be called after the Initialize method to return the negative loglikelihood of the data with respect to the current parameters. 
//	The returned negative loglikelihood is stored in the variable objvalue;
//	The returned gradient vector wrt the current parameters is stored in the variable Gradient; 
void PlainTerminating::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	gradient = Eigen::VectorXd::Zero(num_dims_ * num_dims_);

	Eigen::Map<Eigen::MatrixXd> grad_alpha_matrix = Eigen::Map<Eigen::MatrixXd>(gradient.segment(0, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	Eigen::Map<Eigen::MatrixXd> alpha = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(0, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	objvalue = 0;

	Eigen::VectorXd has_parent_nodes = alpha.colwise().sum();

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		if(has_parent_nodes(i) != 0)
		{
			double local_obj = 0;

			Eigen::VectorXd intensity_per_seq = arrayK[i] * alpha.col(i);
			
			Eigen::VectorXd intensity_integral_per_seq = arrayG[i] * alpha.col(i);
			
			Eigen::VectorXd source_identifier = arrayK[i].rowwise().sum();
			
			for(unsigned c = 0; c < num_sequences_; ++ c)
			{
				if(all_timestamp_per_dimension_[c][i].size() > 0) // if infected
				{
					if((source_identifier(c) != 0) && (intensity_per_seq(c) > 0)) // not a source node
					{
						grad_alpha_matrix.col(i) = grad_alpha_matrix.col(i).array() + arrayK[i].row(c).transpose().array() / intensity_per_seq(c) - arrayG[i].row(c).transpose().array();
				
						local_obj = local_obj + log(intensity_per_seq(c)) - intensity_integral_per_seq(c);

					}else if ((source_identifier(c) != 0) && (intensity_per_seq(c) == 0))
					{
						alpha.col(i) = Eigen::VectorXd::Zero(num_dims_);
					}
				}else // survival
				{

					grad_alpha_matrix.col(i) = grad_alpha_matrix.col(i).array() - arrayG[i].row(c).transpose().array();

					local_obj = local_obj - intensity_integral_per_seq(c);
				}
			}

			objvalue += local_obj;
		}
	}

	gradient = -gradient.array() / num_sequences_;

	objvalue = -objvalue / num_sequences_;

	switch (options_.excitation_regularizer)
	{
		case L1 :

			gradient = gradient.array() + options_.coefficients[LAMBDA];

			objvalue += options_.coefficients[LAMBDA] * parameters_.array().abs().sum();

			return;

		case L22 :

			gradient = gradient.array() + options_.coefficients[LAMBDA] * parameters_.array();

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA] * parameters_.squaredNorm();

			return;

		case NONE :
			return;
	}

}

	//  Return the stochastic gradient on the random sample k.
void PlainTerminating::Gradient(const unsigned &k, Eigen::VectorXd& gradient)
{
	return;
}

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
double PlainTerminating::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
{
	return 0;
}

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
double PlainTerminating::IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
{
	return 0;
}

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the integral of the intensity from a to b
double PlainTerminating::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	return 0;
}

//  This function predicts the next event by simulation;
double PlainTerminating::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	return 0;
}