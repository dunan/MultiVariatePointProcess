#include <cmath>
#include <igl/slice.h>
#include <igl/slice_into.h>
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

	unsigned nVars = process_->GetParameters().size();

	// Eigen::VectorXd x = (Eigen::VectorXd::Random(nVars).array() + 1) * 0.5;

	Eigen::VectorXd x(2);
	x << 0.9, 0.1;

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

	unsigned maxIter = 100;

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

			std::cout << "Halving Step Size" << std::endl;
			t = 0.5 * t;

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
			std::cout << i << " " << funEvals << " " << t << " " << f << " " << 0 << std::endl;
			std::cout << "All variables are at their bound and no further progress is possible" << std::endl;
			break;
		}else 
		{
			igl::slice(g, working, g_working);

			std::cout << i << " " << funEvals << " " << t << " " << f << " " << g_working.array().abs().sum() << std::endl;

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

}

