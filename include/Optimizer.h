#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include "Process.h"
#include "SimpleRNG.h"
#include "RedSVD.h"


class Optimizer
{

private:

	//  Internal implementation for random number generator;
	SimpleRNG RNG_;

	IProcess* process_; 

	double optTol;

	void lbfgs(const Eigen::VectorXd& g, const Eigen::MatrixXd& s, const Eigen::MatrixXd& y, const double& Hdiag, Eigen::VectorXd& d);

	void lbfgsUpdate(const Eigen::VectorXd& y, const Eigen::VectorXd& s, unsigned corrections, Eigen::MatrixXd& old_dirs, Eigen::MatrixXd& old_stps, double& Hdiag);

	bool isLegal(const Eigen::VectorXd& x);

	void ComputeWorkingSet(const Eigen::VectorXd& params, const Eigen::VectorXd& grad, double LB, double UB, Eigen::VectorXi& working);

	void projectBounds(Eigen::VectorXd& params, double LB, double UB);

	unsigned maxIter_;

public:

	Optimizer(IProcess* process) : process_(process)
	{
		RNG_.SetState(0, 0);
		optTol = 1e-10;
		maxIter_ = 500;
	}

	void SGD(const double& gamma0, const unsigned& ini_max_iter, const std::vector<Sequence>& data);

	void PLBFGS(const double& LB, const double& UB);

	void ProximalGroupLasso(const double& gamma0, const double& lambda, const unsigned& ini_max_iter, const unsigned& group_size);

	void ProximalGroupLassoForHawkes(const double& gamma0, const double& lambda, const unsigned& ini_max_iter, const unsigned& group_size);

	void ProximalNuclear(const double& lambda, const double& rho, const unsigned& ini_max_iter, const Eigen::VectorXd& trueparameters);

	void ProximalFrankWolfe(const double& lambda, const double& rho, const unsigned& ini_max_iter, const Eigen::VectorXd& trueparameters);

	void ProximalFrankWolfeForLowRankHawkes(const double& gamma0, const double& lambda0, const double& lambda, const double& ub_lambda0, const double& ub_alpha, const double& rho, const unsigned& ini_max_iter, const unsigned& num_rows, const unsigned& num_cols, const Eigen::MatrixXd& TrueLambda0, const Eigen::MatrixXd& TrueAlpha, const Eigen::VectorXd& X0);

};

#endif