/**
 * \file TerminatingProcessLearningTriggeringKernel.cc
 * \brief The class implementation of TerminatingProcessLearningTriggeringKernel.
 */
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include "../include/TerminatingProcessLearningTriggeringKernel.h"
#include "../include/Utility.h"
#include "../include/GNUPlotWrapper.h"

void TerminatingProcessLearningTriggeringKernel::Initialize(const std::vector<Sequence>& data)
{

	InitializeDimension(data);

	num_sequences_ = data.size();

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		std::vector<Eigen::MatrixXd> MatrixK(num_sequences_, Eigen::MatrixXd::Zero(num_dims_, num_rbfs_));
		std::vector<Eigen::MatrixXd> MatrixG(num_sequences_, Eigen::MatrixXd::Zero(num_dims_, num_rbfs_));

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
						
						double deltaT_ji = i_time - j_time;

						MatrixK[c].row(j) = ((- 0.5) * (((deltaT_ji - tau_.transpose().array()) / sigma_.transpose().array()).square())).exp();

						MatrixG[c].row(j) = (sqrt2PIsigma_.array() * (((tau_.array() - deltaT_ji) / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc)) - erfctau_sigma_.array())).transpose();

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

					double deltaT_ji = data[c].GetTimeWindow() - j_time;

					MatrixG[c].row(j) = (sqrt2PIsigma_.array() * (((tau_.array() - deltaT_ji) / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc)) - erfctau_sigma_.array())).transpose();

				}
			}	
		}

		arrayK.push_back(MatrixK);
		arrayG.push_back(MatrixG);
	}
}

void TerminatingProcessLearningTriggeringKernel::InitializeWithGraph(const std::vector<Sequence>& data)
{

	InitializeDimension(data);

	num_sequences_ = data.size();

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		std::vector<Eigen::MatrixXd> MatrixK(num_sequences_, Eigen::MatrixXd::Zero(num_dims_, num_rbfs_));
		std::vector<Eigen::MatrixXd> MatrixG(num_sequences_, Eigen::MatrixXd::Zero(num_dims_, num_rbfs_));
		
		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			
			if(all_timestamp_per_dimension_[c][i].size() > 0) // if infected
			{
				const double& i_time = all_timestamp_per_dimension_[c][i][0];

				for(std::set<unsigned>::const_iterator i_parent = graph_->nodes[i].parents.begin(); i_parent != graph_->nodes[i].parents.end(); ++ i_parent)
				{
					if((all_timestamp_per_dimension_[c][*i_parent].size() > 0) && (all_timestamp_per_dimension_[c][*i_parent][0] < i_time))
					{
						const int& j = *i_parent;
						const double& j_time = all_timestamp_per_dimension_[c][j][0];

						double deltaT_ji = i_time - j_time;

						MatrixK[c].row(j) = ((- 0.5) * (((deltaT_ji - tau_.transpose().array()) / sigma_.transpose().array()).square())).exp();

						MatrixG[c].row(j) = (sqrt2PIsigma_.array() * (((tau_.array() - deltaT_ji) / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc)) - erfctau_sigma_.array())).transpose();

					}
				}

			}else // if survival
			{
				for(std::set<unsigned>::const_iterator i_parent = graph_->nodes[i].parents.begin(); i_parent != graph_->nodes[i].parents.end(); ++ i_parent)
				{
					if(all_timestamp_per_dimension_[c][*i_parent].size() > 0)
					{
						const int& j = *i_parent;
						const double& j_time = all_timestamp_per_dimension_[c][j][0];

						double deltaT_ji = data[c].GetTimeWindow() - j_time;

						MatrixG[c].row(j) = (sqrt2PIsigma_.array() * (((tau_.array() - deltaT_ji) / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc)) - erfctau_sigma_.array())).transpose();

					}
				}
			}
			
		}

		arrayK.push_back(MatrixK);
		arrayG.push_back(MatrixG);

	}
}

void TerminatingProcessLearningTriggeringKernel::PostProcessing()
{
	double epsilon = 5e-2;
	std::vector<Eigen::Map<Eigen::MatrixXd> > MatrixAlpha;
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		MatrixAlpha.push_back(Eigen::Map<Eigen::MatrixXd>(parameters_.segment(i * num_rbfs_ * num_dims_, num_rbfs_ * num_dims_).data(), num_rbfs_, num_dims_));
	}


	Eigen::MatrixXd Alpha = Eigen::MatrixXd::Zero(num_dims_, num_dims_);
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		MatrixAlpha[i].col(i) = Eigen::VectorXd::Zero(num_rbfs_);
		Eigen::VectorXd parents = Eigen::VectorXd::Zero(num_dims_);
		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			parents = parents.array() + arrayK[i][c].array().abs().rowwise().sum();
		}

		for(unsigned j = 0; j < num_dims_; ++ j)
		{
			if(parents(j) == 0)
			{
				MatrixAlpha[i].col(j) = Eigen::VectorXd::Zero(num_rbfs_);
			}
		}

		for(unsigned j = 0; j < num_dims_; ++ j)
		{
			Alpha(j,i) = MatrixAlpha[i].col(j).norm();
		}

		// std::cout << MatrixAlpha[i] << std::endl << std::endl;

	}

	// std::cout << Alpha << std::endl;

	Eigen::VectorXd colsum = Alpha.colwise().sum();
	colsum = (colsum.array() > 0).select(colsum, 1);
	Alpha = Alpha.array().rowwise() / colsum.transpose().array();
	Alpha = (Alpha.array() < epsilon).select(0, Alpha);
	Alpha = (Alpha.array() >= epsilon).select(1, Alpha);
	std::cout << std::endl << "Recovered Structure" << std::endl << Alpha.cast<unsigned>() << std::endl << std::endl;

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		for(unsigned j = 0; j < num_dims_; ++ j)
		{
			if(Alpha.cast<unsigned>()(i,j) == 0)
			{
				MatrixAlpha[j].col(i) = Eigen::VectorXd::Zero(num_rbfs_);
			}
		}
	}
}

//  MLE esitmation of the parameters
void TerminatingProcessLearningTriggeringKernel::fit(const std::vector<Sequence>& data, const OPTION& options)
{
	if(graph_ == NULL)
	{
		TerminatingProcessLearningTriggeringKernel::Initialize(data);	
	}else
	{
		TerminatingProcessLearningTriggeringKernel::InitializeWithGraph(data);
	}

	options_ = options;
	
	Optimizer opt(this);

	switch (options_.excitation_regularizer)
	{
		case GROUP :

			opt.ProximalGroupLasso(1e-1, options_.coefficients[LAMBDA], 1000, num_rbfs_);

			break;

		default :		
			
			opt.PLBFGS(0, 1e10);
			break;
	}

	if(graph_ == NULL) 
	{
		TerminatingProcessLearningTriggeringKernel::PostProcessing();
	}


	return;
}

void TerminatingProcessLearningTriggeringKernel::GetNegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	gradient = Eigen::VectorXd::Zero(num_rbfs_ * num_dims_ * num_dims_);

	std::vector<Eigen::Map<Eigen::MatrixXd> > MatrixAlpha;
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		MatrixAlpha.push_back(Eigen::Map<Eigen::MatrixXd>(parameters_.segment(i * num_rbfs_ * num_dims_, num_rbfs_ * num_dims_).data(), num_rbfs_, num_dims_));
	}

	std::vector<Eigen::Map<Eigen::MatrixXd> > GradMatrixAlpha;
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		GradMatrixAlpha.push_back(Eigen::Map<Eigen::MatrixXd>(gradient.segment(i * num_rbfs_ * num_dims_, num_rbfs_ * num_dims_).data(), num_rbfs_, num_dims_));
	}

	objvalue = 0;

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		if((MatrixAlpha[i].colwise().sum().array() != 0).any()) // at least has one valid parent
		{
			double local_obj = 0;

			for(unsigned c = 0; c < num_sequences_; ++ c)
			{
				double intensity_c = (arrayK[i][c] * MatrixAlpha[i]).trace() + 1e-4; // to make optimization stable since we do not have a base intensity 

				double intensity_integral_c = (arrayG[i][c] * MatrixAlpha[i]).trace();

				if(all_timestamp_per_dimension_[c][i].size() > 0) // if infected
				{

					Eigen::VectorXd source_identifier = arrayK[i][c].array().abs().rowwise().sum();

					if((source_identifier.array() != 0).any()) // not a source node
					{
						GradMatrixAlpha[i] = GradMatrixAlpha[i].array() + arrayK[i][c].transpose().array() / intensity_c - arrayG[i][c].transpose().array();

						local_obj += (log(intensity_c) - intensity_integral_c);

					}

				}else // survival
				{
					GradMatrixAlpha[i] = GradMatrixAlpha[i].array() - arrayG[i][c].transpose().array();

					local_obj -= intensity_integral_c;
				}
			}

			objvalue += local_obj;

		}
	}

	gradient = -gradient.array() / num_sequences_;

	objvalue = -objvalue / num_sequences_;

}

//  This virtual function requires process-specific implementation. It calculates the negative loglikelihood of the given data. This function must be called after the Initialize method to return the negative loglikelihood of the data with respect to the current parameters. 
//	The returned negative loglikelihood is stored in the variable objvalue;
//	The returned gradient vector wrt the current parameters is stored in the variable Gradient; 
void TerminatingProcessLearningTriggeringKernel::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	GetNegLoglikelihood(objvalue, gradient);
	

	switch (options_.excitation_regularizer)
	{
		case L22 :

			gradient = gradient.array() + options_.coefficients[LAMBDA] * parameters_.array();

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA] * parameters_.squaredNorm();

			return;

		case NONE :
			return;

		default :
			return;
	}
	

}

//  Return the stochastic gradient on the random sample k.
void TerminatingProcessLearningTriggeringKernel::Gradient(const unsigned &k, Eigen::VectorXd& gradient)
{
	return;
}

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
double TerminatingProcessLearningTriggeringKernel::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
{
	return 0;
}

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
double TerminatingProcessLearningTriggeringKernel::IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
{
	return 0;
}

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the integral of the intensity from a to b
double TerminatingProcessLearningTriggeringKernel::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	return 0;
}

//  This function predicts the next event by simulation;
double TerminatingProcessLearningTriggeringKernel::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	return 0;
}

void TerminatingProcessLearningTriggeringKernel::PlotTriggeringKernel(const unsigned& dim_m, const unsigned& dim_n, const double& T, const double& delta)
{
	unsigned num_points = T / delta;
	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(num_points, 0, T);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(num_points);

	std::vector<Eigen::Map<Eigen::MatrixXd> > MatrixAlpha;
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		MatrixAlpha.push_back(Eigen::Map<Eigen::MatrixXd>(parameters_.segment(i * num_rbfs_ * num_dims_, num_rbfs_ * num_dims_).data(), num_rbfs_, num_dims_));
	}

	for(unsigned i = 0; i < num_points; ++ i)
	{
		y(i) = ((-((x(i) - tau_.array()) / sqrt2sigma_.array()).square()).exp()).matrix().transpose() * MatrixAlpha[dim_n].col(dim_m);
	}

	std::vector<double> gp_x(num_points,0);
	std::vector<double> gp_y(num_points,0);

	for(unsigned i = 0; i < num_points; ++ i)
	{
		gp_x[i] = x(i);
		gp_y[i] = y(i);
	}
	
	Plot plot("wxt size 640, 400","time", "intensity");
	plot.PlotScatterLine(gp_x, gp_y, "TriggeringKernel");
	
}
