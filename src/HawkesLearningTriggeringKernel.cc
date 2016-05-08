/**
 * \file HawkesLearningTriggeringKernel.cc
 * \brief The class implementation of HawkesLearningTriggeringKernel.
 */
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include "../include/HawkesLearningTriggeringKernel.h"
#include "../include/GNUPlotWrapper.h"
#include "../include/Utility.h"

void HawkesLearningTriggeringKernel::InitializeConstants()
{
	options_.base_intensity_regularizer = NONE;
	options_.excitation_regularizer = NONE;
	options_.coefficients[LAMBDA0] = 0;
	options_.coefficients[LAMBDA] = 0;
	num_rbfs_ = tau_.size();

	sqrt2sigma_ = sqrt(2) * sigma_.array();
	sqrt2PIsigma_ = 0.5 * sqrt(2 * PI) * sigma_.array();
	erfctau_sigma_ = (tau_.array() / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc));
}

void HawkesLearningTriggeringKernel::Initialize(const std::vector<Sequence>& data)
{
	InitializeDimension(data);

	num_sequences_ = data.size();

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		std::vector<std::vector<Eigen::MatrixXd> > MatrixK(num_sequences_, std::vector<Eigen::MatrixXd>());
		std::vector<Eigen::MatrixXd> MatrixG(num_sequences_, Eigen::MatrixXd::Zero(num_dims_, num_rbfs_));

		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			const std::vector<Event>& seq = data[c].GetEvents();

			const unsigned& nc = all_timestamp_per_dimension_[c][i].size();

			if(nc > 0) // if infected
			{
				MatrixK[c] = std::vector<Eigen::MatrixXd>(nc, Eigen::MatrixXd::Zero(num_dims_, num_rbfs_));

				for(unsigned k = 0; k < nc; ++ k)
				{
					const double& i_time = all_timestamp_per_dimension_[c][i][k];

					for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
					{
						const int& j = i_event->DimentionID;
						const double& j_time = i_event->time;

						if(j_time < i_time)
						{
							double deltaT_ji = i_time - j_time;

							MatrixK[c][k].row(j) = MatrixK[c][k].row(j).array() + ((- 0.5) * (((deltaT_ji - tau_.transpose().array()) / sigma_.transpose().array()).square())).exp();

						}else
						{
							break;
						}
					}
				}
			}

			// survival terms
					
			for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
			{
				const int& j = i_event->DimentionID;
				const double& j_time = i_event->time;

				double deltaT_ji = data[c].GetTimeWindow() - j_time;

				MatrixG[c].row(j) = MatrixG[c].row(j).array() + (sqrt2PIsigma_.array() * (((tau_.array() - deltaT_ji) / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc)) - erfctau_sigma_.array())).transpose();

			}
			
		}

		arrayK.push_back(MatrixK);
		arrayG.push_back(MatrixG);
	}

	observation_window_T_ = Eigen::VectorXd::Zero(num_sequences_);
	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		observation_window_T_(c) = data[c].GetTimeWindow();
	}

}

void HawkesLearningTriggeringKernel::InitializeWithGraph(const std::vector<Sequence>& data)
{
	InitializeDimension(data);

	num_sequences_ = data.size();

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		std::vector<std::vector<Eigen::MatrixXd> > MatrixK(num_sequences_, std::vector<Eigen::MatrixXd>());
		std::vector<Eigen::MatrixXd> MatrixG(num_sequences_, Eigen::MatrixXd::Zero(num_dims_, num_rbfs_));

		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			const std::vector<Event>& seq = data[c].GetEvents();

			const unsigned& nc = all_timestamp_per_dimension_[c][i].size();

			if(nc > 0) // if infected
			{
				MatrixK[c] = std::vector<Eigen::MatrixXd>(nc, Eigen::MatrixXd::Zero(num_dims_, num_rbfs_));

				for(unsigned k = 0; k < nc; ++ k)
				{
					const double& i_time = all_timestamp_per_dimension_[c][i][k];

					for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
					{
						const int& j = i_event->DimentionID;
						const double& j_time = i_event->time;

						if((graph_->nodes[i].parents.find(j) != graph_->nodes[i].parents.end()) && (j_time < i_time))
						{
							
							double deltaT_ji = i_time - j_time;

							MatrixK[c][k].row(j) = MatrixK[c][k].row(j).array() + ((- 0.5) * (((deltaT_ji - tau_.transpose().array()) / sigma_.transpose().array()).square())).exp();

						}else if(j_time >= i_time)
						{
							break;
						}
					}
				}
			}

			// survival terms
			
			for(std::vector<Event>::const_iterator i_event = seq.begin(); i_event != seq.end(); ++ i_event)
			{
				const int& j = i_event->DimentionID;
				const double& j_time = i_event->time;

				if(graph_->nodes[i].parents.find(j) != graph_->nodes[i].parents.end())
				{
					double deltaT_ji = data[c].GetTimeWindow() - j_time;

					MatrixG[c].row(j) = MatrixG[c].row(j).array() + (sqrt2PIsigma_.array() * (((tau_.array() - deltaT_ji) / sqrt2sigma_.array()).unaryExpr(std::ptr_fun(erfc)) - erfctau_sigma_.array())).transpose();
				}
			}

		}

		arrayK.push_back(MatrixK);
		arrayG.push_back(MatrixG);

	}

	observation_window_T_ = Eigen::VectorXd::Zero(num_sequences_);
	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		observation_window_T_(c) = data[c].GetTimeWindow();
	}
}

void HawkesLearningTriggeringKernel::PostProcessing()
{
	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);
	std::cout << Lambda0.transpose() << std::endl;

	std::vector<Eigen::Map<Eigen::MatrixXd> > MatrixAlpha;
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		MatrixAlpha.push_back(Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_ + i * num_rbfs_ * num_dims_, num_rbfs_ * num_dims_).data(), num_rbfs_, num_dims_));
	}

	Eigen::MatrixXd Alpha = Eigen::MatrixXd::Zero(num_dims_, num_dims_);	
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		Eigen::VectorXd parents = Eigen::VectorXd::Zero(num_dims_);
		for(unsigned c = 0; c < num_sequences_; ++ c)
		{
			parents = parents.array() + arrayG[i][c].array().abs().rowwise().sum();
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

	double epsilon = 5e-2;
	Eigen::VectorXd colsum = Alpha.colwise().sum();
	colsum = (colsum.array() > 0).select(colsum, 1);
	Alpha = Alpha.array().rowwise() / colsum.transpose().array();
	Alpha = (Alpha.array() < epsilon).select(0, Alpha);
	Alpha = (Alpha.array() >= epsilon).select(1, Alpha);
	std::cout << Alpha.cast<unsigned>() << std::endl;

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
void HawkesLearningTriggeringKernel::fit(const std::vector<Sequence>& data, const OPTION& options)
{
	if(graph_ == NULL)
	{
		HawkesLearningTriggeringKernel::Initialize(data);	
	}else
	{
		HawkesLearningTriggeringKernel::InitializeWithGraph(data);
	}

	options_ = options;
	
	Optimizer opt(this);

	switch (options_.excitation_regularizer)
	{
		case GROUP :

			opt.ProximalGroupLassoForHawkes(1e-2, options_.coefficients[LAMBDA], 1000, num_rbfs_);

			// break;



		default :			
			opt.PLBFGS(0, 1e10);
			break;
	}

	if(graph_ == NULL)
	{
		HawkesLearningTriggeringKernel::PostProcessing();
	}

	return;

}

void HawkesLearningTriggeringKernel::GetNegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{

	gradient = Eigen::VectorXd::Zero(num_dims_ + num_rbfs_ * num_dims_ * num_dims_);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);
	
	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims_).data(), num_dims_);

	std::vector<Eigen::Map<Eigen::MatrixXd> > MatrixAlpha;
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		MatrixAlpha.push_back(Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_ + i * num_rbfs_ * num_dims_, num_rbfs_ * num_dims_).data(), num_rbfs_, num_dims_));
	}

	std::vector<Eigen::Map<Eigen::MatrixXd> > GradMatrixAlpha;
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		GradMatrixAlpha.push_back(Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims_ + i * num_rbfs_ * num_dims_, num_rbfs_ * num_dims_).data(), num_rbfs_, num_dims_));
	}

	objvalue = 0;

	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		double local_obj = 0;

		for(unsigned c = 0; c < num_sequences_; ++ c)
		{

			const unsigned& nc = all_timestamp_per_dimension_[c][i].size();

			for(unsigned k = 0; k < nc; ++ k)
			{

				double intensity_ck = Lambda0_(i) + (arrayK[i][c][k] * MatrixAlpha[i]).trace() + 1e-4;

				GradMatrixAlpha[i] = GradMatrixAlpha[i].array() + arrayK[i][c][k].transpose().array() / intensity_ck;

				grad_lambda0_vector(i) += (1 / intensity_ck);

				local_obj += log(intensity_ck);

			}
		
			// survival terms

			GradMatrixAlpha[i] = GradMatrixAlpha[i].array() - arrayG[i][c].transpose().array();
			local_obj -= (arrayG[i][c] * MatrixAlpha[i]).trace();
			grad_lambda0_vector(i) -= observation_window_T_(c);
			local_obj -= observation_window_T_(c) * Lambda0_(i);
		}

		objvalue += local_obj;
	}

	gradient = -gradient.array() / num_sequences_;

	objvalue = -objvalue / num_sequences_;

}

//  This virtual function requires process-specific implementation. It calculates the negative loglikelihood of the given data. This function must be called after the Initialize method to return the negative loglikelihood of the data with respect to the current parameters. 
//	The returned negative loglikelihood is stored in the variable objvalue;
//	The returned gradient vector wrt the current parameters is stored in the variable Gradient; 
void HawkesLearningTriggeringKernel::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{
	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	HawkesLearningTriggeringKernel::GetNegLoglikelihood(objvalue, gradient);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);
	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims_).data(), num_dims_);


	// Regularization for base intensity
	switch (options_.base_intensity_regularizer)
	{
		case L22 :
			
			grad_lambda0_vector = grad_lambda0_vector.array() + (options_.coefficients[LAMBDA0] * Lambda0_.array());

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA0] * Lambda0_.squaredNorm();
			
			break;

		case NONE :

			break;

		default:
			break; 	
	}

	// Regularization for excitation matrix
	Eigen::Map<Eigen::VectorXd> grad_alpha_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(num_dims_, num_dims_ * num_rbfs_ * num_dims_).data(), num_dims_ * num_rbfs_ * num_dims_);

	Eigen::Map<Eigen::VectorXd> alpha_vector = Eigen::Map<Eigen::VectorXd>(parameters_.segment(num_dims_, num_dims_ * num_rbfs_ * num_dims_).data(), num_dims_ * num_rbfs_ * num_dims_);

	// Regularization for excitation matrix
	switch (options_.excitation_regularizer)
	{
		case L22 :

			grad_alpha_vector = grad_alpha_vector.array() + options_.coefficients[LAMBDA] * alpha_vector.array();

			objvalue = objvalue + 0.5 * options_.coefficients[LAMBDA] * alpha_vector.squaredNorm();

			return;

		case NONE :
			return;

		default :
			return;
	}

}

//  Return the stochastic gradient on the random sample k.
void HawkesLearningTriggeringKernel::Gradient(const unsigned &k, Eigen::VectorXd& gradient)
{
	return;
}

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
double HawkesLearningTriggeringKernel::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
{
	return 0;
}

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
double HawkesLearningTriggeringKernel::IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
{
	return 0;
}

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the integral of the intensity from a to b
double HawkesLearningTriggeringKernel::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	return 0;
}

//  This function predicts the next event by simulation;
double HawkesLearningTriggeringKernel::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	return 0;
}

void HawkesLearningTriggeringKernel::PlotTriggeringKernel(const unsigned& dim_m, const unsigned& dim_n, const double& T, const double& delta)
{
	unsigned num_points = T / delta;
	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(num_points, 0, T);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(num_points);

	std::vector<Eigen::Map<Eigen::MatrixXd> > MatrixAlpha;
	for(unsigned i = 0; i < num_dims_; ++ i)
	{
		MatrixAlpha.push_back(Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_ + i * num_rbfs_ * num_dims_, num_rbfs_ * num_dims_).data(), num_rbfs_, num_dims_));
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