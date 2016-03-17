#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include "Process.h"
#include "SimpleRNG.h"


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

public:

	Optimizer(IProcess* process) : process_(process)
	{
		RNG_.SetState(0, 0);
		optTol = 1e-10;
	}

	void SGD(const double& gamma0, const unsigned& ini_max_iter, const std::vector<Sequence>& data);

	void PLBFGS(const double& LB, const double& UB);


};

#endif