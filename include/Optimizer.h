// #ifndef OPTIMIZER_H
// #define OPTIMIZER_H

// #include <iostream>
// #include <Eigen/Dense>
// #include <cmath>
// #include <igl/slice.h>
// #include <igl/slice_into.h>
// #include "Process.h"
// #include "SimpleRNG.h"

// class Optimizer
// {
// private:

// 	//  Internal implementation for random number generator;
// 	SimpleRNG RNG_;

// 	IProcess* process_; 

// public:

// 	Optimizer(IProcess* process) : process_(process)
// 	{
// 		RNG_.SetState(0, 0);
// 	}

// 	void SGD(const double& gamma0, const unsigned& ini_max_iter, const std::vector<Sequence>& data);

// 	void PLBFGS(Eigen::VectorXd& x, double LB, double UB);



// };

// #endif