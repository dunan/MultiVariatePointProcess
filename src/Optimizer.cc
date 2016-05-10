/**
 * \file Optimizer.cc
 * \brief The class implementation of Optimizer implementing a collection of optimization algorithms. 
 */
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include "../include/Optimizer.h"
#include "../include/Utility.h"


void Optimizer::SGD(const double& gamma0, const unsigned& ini_max_iter, const std::vector<Sequence>& data)
{
	// const unsigned& num_of_dimensions = process_->GetNumDims();

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

void Optimizer::projectBounds(Eigen::VectorXd& params, double LB, double UB)
{
	params = (params.array() < LB).select(LB, params);
	params = (params.array() > UB).select(UB, params);
}

void Optimizer::ComputeWorkingSet(const Eigen::VectorXd& params, const Eigen::VectorXd& grad, double LB, double UB, Eigen::VectorXi& working)
{	
	Eigen::VectorXi mask = Eigen::VectorXi::Constant(grad.size(), 1);
	mask = ((params.array() < LB + optTol * 2) && (grad.array() >= 0)).select(0, mask);
	mask = ((params.array() > UB - optTol * 2) && (grad.array() <= 0)).select(0, mask);
	std::vector<unsigned> idx;
	for(unsigned i = 0; i < mask.size(); ++ i)
	{
		if(mask(i) == 1)
		{
			idx.push_back(i);
		}
	}
	working = Eigen::VectorXi::Zero(idx.size());
	for(unsigned i = 0; i < idx.size(); ++ i)
	{
		working(i) = idx[i];
	}
}

bool Optimizer::isLegal(const Eigen::VectorXd& x)
{
	for(unsigned i = 0; i < x.size(); ++ i)
	{
		if(std::isnan(x(i)))
		{
			return false;
		}
	}
	return true;
}

void Optimizer::lbfgsUpdate(const Eigen::VectorXd& y, const Eigen::VectorXd& s, unsigned corrections, Eigen::MatrixXd& old_dirs, Eigen::MatrixXd& old_stps, double& Hdiag)
{
	double ys = y.transpose() * s;
	if (ys > 1e-10)
	{
		unsigned numCorrections = old_dirs.cols();

		if(numCorrections < corrections)
		{
			// Full Update
			old_dirs.conservativeResize(Eigen::NoChange, numCorrections + 1);
			old_dirs.col(numCorrections) = s;

			old_stps.conservativeResize(Eigen::NoChange, numCorrections + 1);
			old_stps.col(numCorrections) = y;
		}else
		{
			// Limited-Memory Update
			old_dirs.block(0,0,old_dirs.rows(), numCorrections - 1) = old_dirs.block(0,1,old_dirs.rows(),numCorrections - 1);
			old_dirs.col(numCorrections - 1) = s;

			old_stps.block(0,0,old_stps.rows(), numCorrections - 1) = old_stps.block(0,1,old_stps.rows(),numCorrections - 1);
			old_stps.col(numCorrections - 1) = y;

		}

		Hdiag = ys / (y.transpose() * y);

	}
}

void Optimizer::lbfgs(const Eigen::VectorXd& g, const Eigen::MatrixXd& s, const Eigen::MatrixXd& y, const double& Hdiag, Eigen::VectorXd& d)
{
	unsigned p = s.rows(), k = s.cols();

	Eigen::VectorXd ro(k);

	for(unsigned i = 0; i < k; ++ i)
	{
		ro(i) = 1 / (y.col(i).transpose() * s.col(i));
	}

	Eigen::MatrixXd q = Eigen::MatrixXd::Zero(p, k + 1);
	Eigen::MatrixXd r = Eigen::MatrixXd::Zero(p, k + 1);
	Eigen::VectorXd al = Eigen::VectorXd::Zero(k);
	Eigen::VectorXd be = Eigen::VectorXd::Zero(k);

	q.col(k) = g;

	for(int i = k - 1; i >= 0; --i)
	{
		al(i) = ro(i) * s.col(i).transpose() * q.col(i + 1);
		q.col(i) = q.col(i + 1) - al(i) * y.col(i);
	}

	// Multiply by Initial Hessian
	r.col(0) = Hdiag * q.col(0);

	for(unsigned i = 0; i < k; ++ i)
	{
		be(i) = ro(i) * y.col(i).transpose() * r.col(i);
		r.col(i + 1) = r.col(i) + s.col(i) * (al(i) - be(i));
	}

	d = r.col(k);
}

void Optimizer::PLBFGS(const double& LB, const double& UB)
{

	maxIter_ = 10000;

	std::cout << std::setw(10) << "Iteration" << "\t" << std::setw(10) << "FunEvals" << "\t" << std::setw(10) << "Step Length" << "\t" << std::setw(10) << "Function Val" << "\t" << std::setw(10) << "Opt Cond" << std::endl;
	unsigned nVars = process_->GetParameters().size();

	Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;
	// Eigen::VectorXd x = Eigen::VectorXd::Constant(nVars, 0.1);

	projectBounds(x, LB, UB);

	double f = 0;
	Eigen::VectorXd g;
	process_->SetParameters(x);
	process_->NegLoglikelihood(f, g);

	Eigen::VectorXi working;
	ComputeWorkingSet(x, g, LB, UB, working);

	Eigen::VectorXd g_working;

	// Check Optimality
	if(working.size() == 0)
	{
		std::cout << "All variables are at their bound and no further progress is possible at initial point" << std::endl;
	}else 
	{
		igl::slice(g, working, g_working);

		if(g_working.norm() <= optTol)
		{
			std::cout << "All working variables satisfy optimality condition at initial point" << std::endl;
			return;
		}
	}

	unsigned i = 1, funEvals = 1;

	unsigned maxIter = maxIter_;

	unsigned corrections = 100;
	Eigen::MatrixXd old_dirs(nVars, 0);
	Eigen::MatrixXd old_stps(nVars, 0);
	double Hdiag;
	double suffDec = 1e-4;

	Eigen::VectorXd g_old, x_old, x_new, g_new;

	double f_old, f_new, t;

	while(funEvals < maxIter)
	{
		// Compute Step Direction
		Eigen::VectorXd d = Eigen::VectorXd::Zero(nVars);

		if(i == 1)
		{
			igl::slice_into(g_working, working, d);
			d = -d;
			Hdiag = 1;

		}
		else
		{
			lbfgsUpdate(g - g_old, x - x_old, corrections, old_dirs, old_stps, Hdiag);

			Eigen::MatrixXd old_dirs_working, old_stps_working;

			igl::slice(old_dirs, working, 1, old_dirs_working);
			igl::slice(old_stps, working, 1, old_stps_working);

			Eigen::MatrixXd temp = (old_dirs_working.array() * old_stps_working.array()).colwise().sum();
			std::vector<unsigned> idx;
			for(unsigned k = 0; k < temp.row(0).size(); ++ k)
			{
				if(temp(0, k) > 1e-10)
				{
					idx.push_back(k);
				}
			}
			Eigen::VectorXi working_col = Eigen::VectorXi::Zero(idx.size());
			for(unsigned k = 0; k < idx.size(); ++ k)
			{
				working_col(k) = idx[k];
			}

			igl::slice(old_dirs, working, working_col, old_dirs_working);
			igl::slice(old_stps, working, working_col, old_stps_working);

			Eigen::VectorXd d_working;
			lbfgs(-g_working, old_dirs_working, old_stps_working, Hdiag, d_working);

			igl::slice_into(d_working, working, d);

		}

		g_old = g;
		x_old = x;

		// Check that Progress can be made along the direction
		f_old = f;
		double gtd = g.transpose() * d;
		if(gtd > -optTol)
		{
			std::cout << "Directional Derivative below optTol" << std::endl;
			break;
		}

		// Select Initial Guess to step length
		if(i == 1)
		{
			t = std::min(1 / g_working.array().abs().sum(),1.0);
		}
		else
		{
			t = 1.0;
		}

		// Evaluate the Objective and Projected Gradient at the Initial Step
		x_new = x + t * d;
		projectBounds(x_new, LB, UB);
		process_->SetParameters(x_new);
		process_->NegLoglikelihood(f_new, g_new);
		
		++ funEvals;

		// Backtracking Line Search
		unsigned lineSearchIters = 1;
		while ((f_new > f + suffDec * g.transpose() * (x_new - x)) || std::isnan(f_new))
		{
			double temp = t;

			// std::cout << "Reduce Step Size" << std::endl;
			// t = 0.5 * t;
			t = 0.1 * t;

			// Adjust if change is too small
			if (t < temp * 1e-3)
			{
				std::cout << "Interpolated value too small, Adjusting" << std::endl;
				t = temp * 1e-3;
			}else if(t > temp * 0.6)
			{
				std::cout << "Interpolated value too large, Adjusting" << std::endl;
				t = temp * 0.6;
			}

			// Check whether step has become too small
			if ((t * d).array().abs().sum() < optTol)
			{
				std::cout << "Line Search failed" << std::endl;

				t = 0;
				f_new = f;
				g_new = g;
				break;
			}

			// Evaluate New Point
			x_new = x + t * d;
			projectBounds(x_new, LB, UB);
			process_->SetParameters(x_new);
			process_->NegLoglikelihood(f_new, g_new);

			++ funEvals;
			++ lineSearchIters;
		}

		// Take step
		x = x_new;
		f = f_new;
		g = g_new;

		// Compute working set
		ComputeWorkingSet(x, g, LB, UB, working);

		// Check optimality
		if(working.size() == 0)
		{
			std::cout << std::setw(10) << i << "\t" << std::setw(10) << funEvals << "\t" << std::setw(10) << t << "\t" << std::setw(10) << f << "\t" << std::setw(10) <<  0 << std::endl;
			std::cout << "All variables are at their bound and no further progress is possible" << std::endl;
			break;
		}else 
		{
			igl::slice(g, working, g_working);

			std::cout << std::setw(10) << i << "\t" << std::setw(10) << funEvals << "\t" << std::setw(10) << t << "\t" << std::setw(10) << f << "\t" << std::setw(10) <<  g_working.array().abs().sum() << std::endl;

			if(g_working.norm() <= optTol)
			{
				std::cout << "All working variables satisfy optimality condition" << std::endl;
				break;
			}
		}

		// Check for lack of progress
		if ((t * d).array().abs().sum() < optTol)
		{
			std::cout << "Step size below optTol" << std::endl;
			break;
		}
		
		if (std::fabs(f - f_old) < optTol)
		{
			std::cout << "Function value changing by less than optTol" << std::endl;
			break;	
		}

		if (funEvals > maxIter)
		{
			std::cout << "Function Evaluations exceed maxIter" << std::endl;
			break;		
		}

		++ i;

	}

	std::cout << std::endl;

}

void Optimizer::ProximalGroupLasso(const double& gamma0, const double& lambda, const unsigned& ini_max_iter, const unsigned& group_size)
{

	unsigned nVars = process_->GetParameters().size();

	Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;
	process_->SetParameters(x);

	Eigen::VectorXd gradient;

	unsigned num_dims = process_->GetNumDims();

	std::cout << std::setw(10) << "Iteration" << "\t" << std::setw(10) << "Step Length" << "\t" << std::setw(10) << "Function Val" << "\t" << std::setw(10) << "Opt Cond" << std::endl;

	double threshold = gamma0 * lambda;

	double f_old, f_new;
	process_->NegLoglikelihood(f_old, gradient);

	std::cout << std::setw(10) << 0 << "\t" << std::setw(10) << gamma0 << "\t" << std::setw(10) << f_old << "\t" << std::setw(10) <<  gradient.array().abs().sum() << std::endl;

	for(unsigned iter = 1; iter < ini_max_iter; ++ iter)
	{
		
		x = process_->GetParameters();

		for(unsigned i = 0; i < num_dims; ++ i)
		{
			Eigen::Map<Eigen::MatrixXd> MatrixAlpha = Eigen::Map<Eigen::MatrixXd>(x.segment(i * group_size * num_dims, group_size * num_dims).data(), group_size, num_dims);
			
			Eigen::Map<Eigen::MatrixXd> GradMatrixAlpha = Eigen::Map<Eigen::MatrixXd>(gradient.segment(i * group_size * num_dims, group_size * num_dims).data(), group_size, num_dims);

			for(unsigned j = 0; j < num_dims; ++ j)
			{
				Eigen::VectorXd valid_group_identifier = MatrixAlpha.array().abs().colwise().sum();
				if(valid_group_identifier(j) != 0)
				{

					MatrixAlpha.col(j) = MatrixAlpha.col(j) - gamma0 * GradMatrixAlpha.col(j);

					// proximal mapping
					if(MatrixAlpha.col(j).norm() > threshold)
					{
						// first shrink it
						MatrixAlpha.col(j) = MatrixAlpha.col(j) - threshold * MatrixAlpha.col(j).normalized();

						// then, make projections
						MatrixAlpha.col(j) = (MatrixAlpha.col(j).array() > 0).select(MatrixAlpha.col(j), 0);

					}else
					{
						MatrixAlpha.col(j) = Eigen::VectorXd::Zero(group_size);
					}
				}
			}
		}

		process_->SetParameters(x);
		process_->NegLoglikelihood(f_new, gradient);

		if (std::fabs(f_new - f_old) < optTol)
		{
			std::cout << "Function value changing by less than optTol" << std::endl;
			break;	
		}

		std::cout << std::setw(10) << iter << "\t" << std::setw(10) << gamma0 << "\t" << std::setw(10) << f_new << "\t" << std::setw(10) <<  gradient.array().abs().sum() << std::endl;

		f_old = f_new;

	}

}

void ProximalMapping(const double& gamma0, const double& threshold, const unsigned& num_dims, const unsigned& group_size, Eigen::VectorXd& x, Eigen::VectorXd& gradient, Eigen::VectorXd& x_new)
{
	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(x.segment(0, num_dims).data(), num_dims);

	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims).data(), num_dims);

	Eigen::Map<Eigen::VectorXd> Lambda0_new = Eigen::Map<Eigen::VectorXd>(x_new.segment(0, num_dims).data(), num_dims);

	// update the base intensity by proximal gradient
	Lambda0_new = Lambda0.array() - gamma0 * grad_lambda0_vector.array();

	// then, make projections
	Lambda0_new = (Lambda0.array() > 0).select(Lambda0, 0);

	// Upate excitation matrices
	for(unsigned i = 0; i < num_dims; ++ i)
	{
		Eigen::Map<Eigen::MatrixXd> MatrixAlpha = Eigen::Map<Eigen::MatrixXd>(x.segment(num_dims + i * group_size * num_dims, group_size * num_dims).data(), group_size, num_dims);

		Eigen::Map<Eigen::MatrixXd> MatrixAlpha_new = Eigen::Map<Eigen::MatrixXd>(x_new.segment(num_dims + i * group_size * num_dims, group_size * num_dims).data(), group_size, num_dims);

		MatrixAlpha_new = Eigen::MatrixXd::Zero(group_size, num_dims);
		
		Eigen::Map<Eigen::MatrixXd> GradMatrixAlpha = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims + i * group_size * num_dims, group_size * num_dims).data(), group_size, num_dims);

		for(unsigned j = 0; j < num_dims; ++ j)
		{
			Eigen::VectorXd valid_group_identifier = MatrixAlpha.array().abs().colwise().sum();
			if(valid_group_identifier(j) != 0)
			{

				MatrixAlpha_new.col(j) = MatrixAlpha.col(j) - gamma0 * GradMatrixAlpha.col(j);

				// proximal mapping
				if(MatrixAlpha_new.col(j).norm() > threshold)
				{
					// first shrink it
					MatrixAlpha_new.col(j) = MatrixAlpha_new.col(j) - threshold * MatrixAlpha_new.col(j).normalized();

					// then, make projections
					MatrixAlpha_new.col(j) = (MatrixAlpha_new.col(j).array() > 0).select(MatrixAlpha_new.col(j), 0);

				}else
				{
					MatrixAlpha_new.col(j) = Eigen::VectorXd::Zero(group_size);
				}
			}
		}
	}
}

void Optimizer::ProximalGroupLassoForHawkes(const double& gamma_ini, const double& lambda, const unsigned& ini_max_iter, const unsigned& group_size)
{
	unsigned nVars = process_->GetParameters().size();

	Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;
	process_->SetParameters(x);

	Eigen::VectorXd g, g_new;

	unsigned num_dims = process_->GetNumDims();

	std::cout << std::setw(10) << "Iteration" << "\t" << std::setw(10) << "Step Length" << "\t" << std::setw(10) << "Function Val" << "\t" << std::setw(10) << "Opt Cond" << std::endl;

	double f, f_new;
	process_->NegLoglikelihood(f, g);

	std::cout << std::setw(10) << 0 << "\t" << std::setw(10) << gamma_ini << "\t" << std::setw(10) << f << "\t" << std::setw(10) <<  g.array().abs().sum() << std::endl;

	Eigen::VectorXd x_new = Eigen::VectorXd::Zero(nVars);

	double suffDec = 1e-4;

	for(unsigned iter = 1; iter < ini_max_iter; ++ iter)
	{
		// Select Initial Guess to step length
		double gamma0 = gamma_ini;

		ProximalMapping(gamma0, gamma0 * lambda, num_dims, group_size, x, g, x_new);
		process_->SetParameters(x_new);
		process_->NegLoglikelihood(f_new, g_new);

		// Backtracking Line Search
		while ((f_new > f + suffDec * g.transpose() * (x_new - x)) || std::isnan(f_new))
		{

			std::cout << "Reduce Step Size" << std::endl;

			gamma0 = 0.1 * gamma0;

			ProximalMapping(gamma0, gamma0 * lambda, num_dims, group_size, x, g, x_new);
			process_->SetParameters(x_new);
			process_->NegLoglikelihood(f_new, g_new);
		}

		std::cout << std::setw(10) << iter << "\t" << std::setw(10) << gamma0 << "\t" << std::setw(10) << f_new << "\t" << std::setw(10) <<  g_new.array().abs().sum() << std::endl;

		if (std::fabs(f_new - f) < optTol)
		{
			std::cout << "Function value changing by less than optTol" << std::endl;
			break;	
		}

		x = x_new;
		f = f_new;
		g = g_new;
	}
}



void Optimizer::ProximalNuclear(const double& lambda, const double& rho, const unsigned& ini_max_iter, const Eigen::VectorXd& trueparameters)
{
	unsigned nVars = process_->GetParameters().size();

	Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;
	process_->SetParameters(x);

	Eigen::VectorXd gradient;

	unsigned num_dims = process_->GetNumDims();

	double f_old, f_new;
	process_->NegLoglikelihood(f_old, gradient);

	Eigen::Map<Eigen::VectorXd> Y_Lambda0 = Eigen::Map<Eigen::VectorXd>(x.segment(0, num_dims).data(), num_dims);
	Eigen::Map<Eigen::VectorXd> Y_grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims).data(), num_dims);
	Eigen::Map<Eigen::MatrixXd> Y_MatrixAlpha = Eigen::Map<Eigen::MatrixXd>(x.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
	Eigen::Map<Eigen::MatrixXd> Y_GradMatrixAlpha = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
	Eigen::MatrixXd Y_Z = Y_MatrixAlpha;

	Eigen::VectorXd X_Lambda0 = Y_Lambda0;
	Eigen::MatrixXd X_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd X_Z = X_MatrixAlpha;

	Eigen::VectorXd U_Lambda0 = Y_Lambda0;
	Eigen::MatrixXd U_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd U_Z = U_MatrixAlpha;

	for(unsigned iter = 1; iter < ini_max_iter; ++ iter)
	{

		double eta = std::min(1e-5 * (iter + 1), 10.0);
		double delta = 2.0 / double(iter + 1);

		Y_Lambda0 = (1 - delta) * X_Lambda0.array() + delta * U_Lambda0.array();
		Y_MatrixAlpha = (1 - delta) * X_MatrixAlpha.array() + delta * U_MatrixAlpha.array();
		Y_Z = (1 - delta) * X_Z.array() + delta * U_Z.array();

		process_->SetParameters(x);
		process_->NegLoglikelihood(f_new, gradient);
		Y_GradMatrixAlpha = Y_GradMatrixAlpha.array() + rho * (Y_MatrixAlpha.array() - Y_Z.array());

		// Proximal Update
		U_Lambda0 = Y_Lambda0.array() - eta * Y_grad_lambda0_vector.array();
		U_Lambda0 = (U_Lambda0.array() > 0).select(U_Lambda0, 0);

		U_MatrixAlpha = Y_MatrixAlpha.array() - eta * Y_GradMatrixAlpha.array();
		U_MatrixAlpha = (U_MatrixAlpha.array() > 0).select(U_MatrixAlpha, 0);

		X_Lambda0 = (1 - delta) * X_Lambda0.array() + delta * U_Lambda0.array();
		X_MatrixAlpha = (1 - delta) * X_MatrixAlpha.array() + delta * U_MatrixAlpha.array();

		// Proximal Update for Z
		Eigen::JacobiSVD<Eigen::MatrixXd> svdfull(Y_Z.array() + eta * rho * (Y_MatrixAlpha.array() - Y_Z.array()),Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::VectorXd singular_values = svdfull.singularValues();
		singular_values = singular_values.array() - eta * lambda;
		U_Z = svdfull.matrixU() * singular_values.asDiagonal() * svdfull.matrixV().transpose();

		X_Z = (1 - delta) * X_Z.array() + delta * U_Z.array();

		Eigen::VectorXd x_new(nVars);
		Eigen::Map<Eigen::VectorXd> Lambda0_new = Eigen::Map<Eigen::VectorXd>(x_new.segment(0, num_dims).data(), num_dims);
		Eigen::Map<Eigen::MatrixXd> MatrixAlpha_new = Eigen::Map<Eigen::MatrixXd>(x_new.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
		Lambda0_new = X_Lambda0;
		MatrixAlpha_new = X_MatrixAlpha;

		std::cout << std::setw(10) << iter << "\t" << std::setw(10) << delta << "\t" << std::setw(10) << f_new << "\t" << std::setw(10) <<  gradient.array().abs().sum() <<"\t" << (x_new - trueparameters).array().abs().mean() << std::endl;
	}
}

void Optimizer::ProximalFrankWolfe(const double& gamma0, const double& lambda, const double& rho, const double& ub_alpha, const unsigned& ini_max_iter, const Eigen::VectorXd& trueparameters)
{
	unsigned nVars = process_->GetParameters().size();

	Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;
	process_->SetParameters(x);

	Eigen::VectorXd gradient;

	unsigned num_dims = process_->GetNumDims();

	double f_old, f_new;
	process_->NegLoglikelihood(f_old, gradient);

	Eigen::Map<Eigen::VectorXd> Y_Lambda0 = Eigen::Map<Eigen::VectorXd>(x.segment(0, num_dims).data(), num_dims);
	Eigen::Map<Eigen::VectorXd> Y_grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims).data(), num_dims);
	Eigen::Map<Eigen::MatrixXd> Y_MatrixAlpha = Eigen::Map<Eigen::MatrixXd>(x.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
	Eigen::Map<Eigen::MatrixXd> Y_GradMatrixAlpha = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
	Eigen::MatrixXd Y_Z = Y_MatrixAlpha;

	Eigen::VectorXd X_Lambda0 = Y_Lambda0;
	Eigen::MatrixXd X_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd X_Z = X_MatrixAlpha;

	Eigen::VectorXd U_Lambda0 = Y_Lambda0;
	Eigen::MatrixXd U_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd U_Z = U_MatrixAlpha;

	for(unsigned iter = 1; iter < ini_max_iter; ++ iter)
	{

		// double eta = std::min(5e-5 * (iter + 1), 10.0);
		double eta = std::min(gamma0 * (iter + 1), 10.0);
		double delta = 2.0 / double(iter + 1);

		Y_Lambda0 = (1 - delta) * X_Lambda0.array() + delta * U_Lambda0.array();
		Y_MatrixAlpha = (1 - delta) * X_MatrixAlpha.array() + delta * U_MatrixAlpha.array();
		Y_Z = (1 - delta) * X_Z.array() + delta * U_Z.array();

		process_->SetParameters(x);
		process_->NegLoglikelihood(f_new, gradient);
		Y_GradMatrixAlpha = Y_GradMatrixAlpha.array() + rho * (Y_MatrixAlpha.array() - Y_Z.array());

		// Proximal Update
		U_Lambda0 = Y_Lambda0.array() - eta * Y_grad_lambda0_vector.array();
		U_Lambda0 = (U_Lambda0.array() > 0).select(U_Lambda0, 0);

		U_MatrixAlpha = Y_MatrixAlpha.array() - eta * Y_GradMatrixAlpha.array();
		U_MatrixAlpha = (U_MatrixAlpha.array() > 0).select(U_MatrixAlpha, 0);

		X_Lambda0 = (1 - delta) * X_Lambda0.array() + delta * U_Lambda0.array();
		X_MatrixAlpha = (1 - delta) * X_MatrixAlpha.array() + delta * U_MatrixAlpha.array();

		// FrankWolfe Update for Z
		Eigen::VectorXd u;
		Eigen::VectorXd v;
		PowerMethod(rho * (Y_MatrixAlpha - Y_Z), 300, 1e-6, u, v);
		U_Z = u * v.transpose();

		double alpha_Z = (rho * ((Y_MatrixAlpha.array() - (1 - delta) * Y_Z.array()) * U_Z.array()).sum() - lambda) / (rho * U_Z.squaredNorm());
		alpha_Z = std::fmin(ub_alpha, std::fmax(alpha_Z / delta, 0.0));

		U_Z = alpha_Z * U_Z.array();
		
		X_Z = (1 - delta) * X_Z.array() + delta * U_Z.array();

		Eigen::VectorXd x_new(nVars);
		Eigen::Map<Eigen::VectorXd> Lambda0_new = Eigen::Map<Eigen::VectorXd>(x_new.segment(0, num_dims).data(), num_dims);
		Eigen::Map<Eigen::MatrixXd> MatrixAlpha_new = Eigen::Map<Eigen::MatrixXd>(x_new.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
		Lambda0_new = X_Lambda0;
		MatrixAlpha_new = X_MatrixAlpha;

		std::cout << std::setw(10) << iter << "\t" << std::setw(10) << delta << "\t" << std::setw(10) << f_new << "\t" << std::setw(10) <<  gradient.array().abs().sum() <<"\t" << (x_new - trueparameters).array().abs().mean() << std::endl;
	}	
}

void Optimizer::ProximalFrankWolfe(const double& gamma0, const double& lambda, const double& rho, const double& ub_alpha, const unsigned& ini_max_iter)
{
	unsigned nVars = process_->GetParameters().size();

	Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;
	process_->SetParameters(x);

	Eigen::VectorXd gradient;

	unsigned num_dims = process_->GetNumDims();

	double f_old, f_new;
	process_->NegLoglikelihood(f_old, gradient);

	Eigen::Map<Eigen::VectorXd> Y_Lambda0 = Eigen::Map<Eigen::VectorXd>(x.segment(0, num_dims).data(), num_dims);
	Eigen::Map<Eigen::VectorXd> Y_grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims).data(), num_dims);
	Eigen::Map<Eigen::MatrixXd> Y_MatrixAlpha = Eigen::Map<Eigen::MatrixXd>(x.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
	Eigen::Map<Eigen::MatrixXd> Y_GradMatrixAlpha = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
	Eigen::MatrixXd Y_Z = Y_MatrixAlpha;

	Eigen::VectorXd X_Lambda0 = Y_Lambda0;
	Eigen::MatrixXd X_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd X_Z = X_MatrixAlpha;

	Eigen::VectorXd U_Lambda0 = Y_Lambda0;
	Eigen::MatrixXd U_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd U_Z = U_MatrixAlpha;

	for(unsigned iter = 1; iter < ini_max_iter; ++ iter)
	{

		// double eta = std::min(5e-5 * (iter + 1), 10.0);
		double eta = std::min(gamma0 * (iter + 1), 10.0);
		double delta = 2.0 / double(iter + 1);

		Y_Lambda0 = (1 - delta) * X_Lambda0.array() + delta * U_Lambda0.array();
		Y_MatrixAlpha = (1 - delta) * X_MatrixAlpha.array() + delta * U_MatrixAlpha.array();
		Y_Z = (1 - delta) * X_Z.array() + delta * U_Z.array();

		process_->SetParameters(x);
		process_->NegLoglikelihood(f_new, gradient);
		Y_GradMatrixAlpha = Y_GradMatrixAlpha.array() + rho * (Y_MatrixAlpha.array() - Y_Z.array());

		// Proximal Update
		U_Lambda0 = Y_Lambda0.array() - eta * Y_grad_lambda0_vector.array();
		U_Lambda0 = (U_Lambda0.array() > 0).select(U_Lambda0, 0);

		U_MatrixAlpha = Y_MatrixAlpha.array() - eta * Y_GradMatrixAlpha.array();
		U_MatrixAlpha = (U_MatrixAlpha.array() > 0).select(U_MatrixAlpha, 0);

		X_Lambda0 = (1 - delta) * X_Lambda0.array() + delta * U_Lambda0.array();
		X_MatrixAlpha = (1 - delta) * X_MatrixAlpha.array() + delta * U_MatrixAlpha.array();

		// FrankWolfe Update for Z
		Eigen::VectorXd u;
		Eigen::VectorXd v;
		PowerMethod(rho * (Y_MatrixAlpha - Y_Z), 300, 1e-6, u, v);
		U_Z = u * v.transpose();

		double alpha_Z = (rho * ((Y_MatrixAlpha.array() - (1 - delta) * Y_Z.array()) * U_Z.array()).sum() - lambda) / (rho * U_Z.squaredNorm());
		alpha_Z = std::fmin(ub_alpha, std::fmax(alpha_Z / delta, 0.0));

		U_Z = alpha_Z * U_Z.array();
		
		X_Z = (1 - delta) * X_Z.array() + delta * U_Z.array();

		Eigen::VectorXd x_new(nVars);
		Eigen::Map<Eigen::VectorXd> Lambda0_new = Eigen::Map<Eigen::VectorXd>(x_new.segment(0, num_dims).data(), num_dims);
		Eigen::Map<Eigen::MatrixXd> MatrixAlpha_new = Eigen::Map<Eigen::MatrixXd>(x_new.segment(num_dims, num_dims * num_dims).data(), num_dims, num_dims);
		Lambda0_new = X_Lambda0;
		MatrixAlpha_new = X_MatrixAlpha;

		std::cout << std::setw(10) << iter << "\t" << std::setw(10) << delta << "\t" << std::setw(10) << f_new << "\t" << std::setw(10) <<  gradient.array().abs().sum() << std::endl;
	}	
}

void Optimizer::ProximalFrankWolfeForLowRankHawkes(const double& gamma0, const double& lambda0, const double& lambda, const double& ub_lambda0, const double& ub_alpha, const double& rho, const unsigned& ini_max_iter, const unsigned& num_rows, const unsigned& num_cols, const Eigen::MatrixXd& TrueLambda0, const Eigen::MatrixXd& TrueAlpha)
{
	unsigned nVars = process_->GetParameters().size();

	Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;
	process_->SetParameters(x);

	Eigen::VectorXd gradient;

	unsigned num_dims = process_->GetNumDims();

	double f_old, f_new;
	process_->NegLoglikelihood(f_old, gradient);

	Eigen::Map<Eigen::MatrixXd> Y_MatrixLambda0 = Eigen::Map<Eigen::MatrixXd>(x.segment(0, num_dims).data(), num_rows, num_cols);	
	Eigen::Map<Eigen::MatrixXd> Y_GradMatrixLambda0 = Eigen::Map<Eigen::MatrixXd>(gradient.segment(0, num_dims).data(), num_rows, num_cols);
	Eigen::Map<Eigen::MatrixXd> Y_MatrixAlpha = Eigen::Map<Eigen::MatrixXd>(x.segment(num_dims, num_dims).data(), num_rows, num_cols);
	Eigen::Map<Eigen::MatrixXd> Y_GradMatrixAlpha = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims, num_dims).data(), num_rows, num_cols);
	Eigen::MatrixXd Y_Z1 = Y_MatrixLambda0;
	Eigen::MatrixXd Y_Z2 = Y_MatrixAlpha;

	Eigen::MatrixXd X_MatrixLambda0 = Y_MatrixLambda0;
	Eigen::MatrixXd X_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd X_Z1 = X_MatrixLambda0;
	Eigen::MatrixXd X_Z2 = X_MatrixAlpha;

	Eigen::MatrixXd U_MatrixLambda0 = Y_MatrixLambda0;
	Eigen::MatrixXd U_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd U_Z1 = U_MatrixLambda0;
	Eigen::MatrixXd U_Z2 = U_MatrixAlpha;

	std::cout << std::setw(10) << "Iteration" << "\t" << std::setw(10) << "Step Length" << "\t" << std::setw(10) << "Function Val" << "\t" << std::setw(20) << "Base intensity MAE" << "\t" << std::setw(20) << "Excitation matrix MAE" << std::endl;
	for(unsigned iter = 1; iter <= ini_max_iter; ++ iter)
	{

		double eta = std::min(gamma0 * (iter + 1), 10.0);
		double delta = 2.0 / double(iter + 1);

		Y_MatrixLambda0 = (1 - delta) * X_MatrixLambda0 + delta * U_MatrixLambda0;
		Y_MatrixAlpha = (1 - delta) * X_MatrixAlpha + delta * U_MatrixAlpha;
		Y_Z1 = (1 - delta) * X_Z1 + delta * U_Z1;
		Y_Z2 = (1 - delta) * X_Z2 + delta * U_Z2;

		process_->SetParameters(x);
		process_->NegLoglikelihood(f_new, gradient);
		Y_GradMatrixLambda0 = Y_GradMatrixLambda0 + rho * (Y_MatrixLambda0 - Y_Z1);
		Y_GradMatrixAlpha = Y_GradMatrixAlpha + rho * (Y_MatrixAlpha - Y_Z2);

		// Proximal Update
		U_MatrixLambda0 = Y_MatrixLambda0 - eta * Y_GradMatrixLambda0;
		U_MatrixLambda0 = (U_MatrixLambda0.array() > 0).select(U_MatrixLambda0, 0);
		U_MatrixAlpha = Y_MatrixAlpha - eta * Y_GradMatrixAlpha;
		U_MatrixAlpha = (U_MatrixAlpha.array() > 0).select(U_MatrixAlpha, 0);

		X_MatrixLambda0 = (1 - delta) * X_MatrixLambda0 + delta * U_MatrixLambda0;
		X_MatrixAlpha = (1 - delta) * X_MatrixAlpha + delta * U_MatrixAlpha;

		// FrankWolfe Update for Z
		// Eigen::JacobiSVD<Eigen::MatrixXd> svdfull(rho * (Y_MatrixLambda0 - Y_Z1), Eigen::ComputeThinU | Eigen::ComputeThinV);
		// U_Z1 = svdfull.matrixU().col(0) * svdfull.matrixV().col(0).transpose();
		// Eigen::JacobiSVD<Eigen::MatrixXd> svdfull2(rho * (Y_MatrixAlpha - Y_Z2), Eigen::ComputeThinU | Eigen::ComputeThinV);
		// U_Z2 = svdfull2.matrixU().col(0) * svdfull2.matrixV().col(0).transpose();

		Eigen::VectorXd u;
		Eigen::VectorXd v;

		PowerMethod(rho * (Y_MatrixLambda0 - Y_Z1), 300, 1e-6, u, v);
		U_Z1 = u * v.transpose();
		PowerMethod(rho * (Y_MatrixAlpha - Y_Z2), 300, 1e-6, u, v);
		U_Z2 = u * v.transpose();

		double alpha_Z1 = (rho * ((Y_MatrixLambda0.array() - (1 - delta) * Y_Z1.array()) * U_Z1.array()).sum() - lambda0) / (rho * U_Z1.squaredNorm());
		double alpha_Z2 = (rho * ((Y_MatrixAlpha.array() - (1 - delta) * Y_Z2.array()) * U_Z2.array()).sum() - lambda) / (rho * U_Z2.squaredNorm());
		alpha_Z1 = std::fmin(ub_lambda0, std::fmax(alpha_Z1 / delta, 0.0));
		alpha_Z2 = std::fmin(ub_alpha, std::fmax(alpha_Z2 / delta, 0.0));

		U_Z1 = alpha_Z1 * U_Z1;
		U_Z2 = alpha_Z2 * U_Z2;

		X_Z1 = (1 - delta) * X_Z1 + delta * U_Z1;
		X_Z2 = (1 - delta) * X_Z2 + delta * U_Z2;

		Eigen::VectorXd x_new(nVars);
		Eigen::Map<Eigen::MatrixXd> MatrixLambda0_new = Eigen::Map<Eigen::MatrixXd>(x_new.segment(0, num_dims).data(), num_rows, num_cols);
		Eigen::Map<Eigen::MatrixXd> MatrixAlpha_new = Eigen::Map<Eigen::MatrixXd>(x_new.segment(num_dims, num_dims).data(), num_rows, num_cols);
		MatrixLambda0_new = X_MatrixLambda0;
		MatrixAlpha_new = X_MatrixAlpha;

		std::cout << std::setw(10) << iter << "\t" << std::setw(10) << delta << "\t" << std::setw(10) << f_new << "\t" << std::setw(20)  << (MatrixLambda0_new - TrueLambda0).array().abs().mean() << "\t" << std::setw(20) << (MatrixAlpha_new - TrueAlpha).array().abs().mean() << std::endl;
	}
}

void Optimizer::ProximalFrankWolfeForLowRankHawkes(const double& gamma0, const double& lambda0, const double& lambda, const double& ub_lambda0, const double& ub_alpha, const double& rho, const unsigned& ini_max_iter, const unsigned& num_rows, const unsigned& num_cols)
{
	unsigned nVars = process_->GetParameters().size();

	Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;
	process_->SetParameters(x);

	Eigen::VectorXd gradient;

	unsigned num_dims = process_->GetNumDims();

	double f_old, f_new;
	process_->NegLoglikelihood(f_old, gradient);

	Eigen::Map<Eigen::MatrixXd> Y_MatrixLambda0 = Eigen::Map<Eigen::MatrixXd>(x.segment(0, num_dims).data(), num_rows, num_cols);	
	Eigen::Map<Eigen::MatrixXd> Y_GradMatrixLambda0 = Eigen::Map<Eigen::MatrixXd>(gradient.segment(0, num_dims).data(), num_rows, num_cols);
	Eigen::Map<Eigen::MatrixXd> Y_MatrixAlpha = Eigen::Map<Eigen::MatrixXd>(x.segment(num_dims, num_dims).data(), num_rows, num_cols);
	Eigen::Map<Eigen::MatrixXd> Y_GradMatrixAlpha = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims, num_dims).data(), num_rows, num_cols);
	Eigen::MatrixXd Y_Z1 = Y_MatrixLambda0;
	Eigen::MatrixXd Y_Z2 = Y_MatrixAlpha;

	Eigen::MatrixXd X_MatrixLambda0 = Y_MatrixLambda0;
	Eigen::MatrixXd X_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd X_Z1 = X_MatrixLambda0;
	Eigen::MatrixXd X_Z2 = X_MatrixAlpha;

	Eigen::MatrixXd U_MatrixLambda0 = Y_MatrixLambda0;
	Eigen::MatrixXd U_MatrixAlpha = Y_MatrixAlpha;
	Eigen::MatrixXd U_Z1 = U_MatrixLambda0;
	Eigen::MatrixXd U_Z2 = U_MatrixAlpha;

	std::cout << std::setw(10) << "Iteration" << "\t" << std::setw(10) << "Step Length" << "\t" << std::setw(10) << "Function Val" << std::endl;
	for(unsigned iter = 1; iter <= ini_max_iter; ++ iter)
	{

		double eta = std::min(gamma0 * (iter + 1), 10.0);
		double delta = 2.0 / double(iter + 1);

		Y_MatrixLambda0 = (1 - delta) * X_MatrixLambda0 + delta * U_MatrixLambda0;
		Y_MatrixAlpha = (1 - delta) * X_MatrixAlpha + delta * U_MatrixAlpha;
		Y_Z1 = (1 - delta) * X_Z1 + delta * U_Z1;
		Y_Z2 = (1 - delta) * X_Z2 + delta * U_Z2;

		process_->SetParameters(x);
		process_->NegLoglikelihood(f_new, gradient);
		Y_GradMatrixLambda0 = Y_GradMatrixLambda0 + rho * (Y_MatrixLambda0 - Y_Z1);
		Y_GradMatrixAlpha = Y_GradMatrixAlpha + rho * (Y_MatrixAlpha - Y_Z2);

		// Proximal Update
		U_MatrixLambda0 = Y_MatrixLambda0 - eta * Y_GradMatrixLambda0;
		U_MatrixLambda0 = (U_MatrixLambda0.array() > 0).select(U_MatrixLambda0, 0);
		U_MatrixAlpha = Y_MatrixAlpha - eta * Y_GradMatrixAlpha;
		U_MatrixAlpha = (U_MatrixAlpha.array() > 0).select(U_MatrixAlpha, 0);

		X_MatrixLambda0 = (1 - delta) * X_MatrixLambda0 + delta * U_MatrixLambda0;
		X_MatrixAlpha = (1 - delta) * X_MatrixAlpha + delta * U_MatrixAlpha;

		Eigen::VectorXd u;
		Eigen::VectorXd v;

		PowerMethod(rho * (Y_MatrixLambda0 - Y_Z1), 300, 1e-6, u, v);
		U_Z1 = u * v.transpose();
		PowerMethod(rho * (Y_MatrixAlpha - Y_Z2), 300, 1e-6, u, v);
		U_Z2 = u * v.transpose();

		double alpha_Z1 = (rho * ((Y_MatrixLambda0.array() - (1 - delta) * Y_Z1.array()) * U_Z1.array()).sum() - lambda0) / (rho * U_Z1.squaredNorm());
		double alpha_Z2 = (rho * ((Y_MatrixAlpha.array() - (1 - delta) * Y_Z2.array()) * U_Z2.array()).sum() - lambda) / (rho * U_Z2.squaredNorm());
		alpha_Z1 = std::fmin(ub_lambda0, std::fmax(alpha_Z1 / delta, 0.0));
		alpha_Z2 = std::fmin(ub_alpha, std::fmax(alpha_Z2 / delta, 0.0));

		U_Z1 = alpha_Z1 * U_Z1;
		U_Z2 = alpha_Z2 * U_Z2;

		X_Z1 = (1 - delta) * X_Z1 + delta * U_Z1;
		X_Z2 = (1 - delta) * X_Z2 + delta * U_Z2;

		Eigen::VectorXd x_new(nVars);
		Eigen::Map<Eigen::MatrixXd> MatrixLambda0_new = Eigen::Map<Eigen::MatrixXd>(x_new.segment(0, num_dims).data(), num_rows, num_cols);
		Eigen::Map<Eigen::MatrixXd> MatrixAlpha_new = Eigen::Map<Eigen::MatrixXd>(x_new.segment(num_dims, num_dims).data(), num_rows, num_cols);
		MatrixLambda0_new = X_MatrixLambda0;
		MatrixAlpha_new = X_MatrixAlpha;

		std::cout << std::setw(10) << iter << "\t" << std::setw(10) << delta << "\t" << std::setw(10) << f_new << std::endl;
	}
}
