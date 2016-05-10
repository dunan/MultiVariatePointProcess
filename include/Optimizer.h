/**
 * \file Optimizer.h
 * \brief The class definition of Optimizer implementing a collection of optimization algorithms. 
 */
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include "Process.h"
#include "SimpleRNG.h"
#include "Utility.h"

/**
 * \class Optimizer Optimizer.h "include/Optimizer.h"
 * \brief Optimizer encapsulates a collection of optimization algorithms used in the toolbox.
 */
class Optimizer
{

private:

/**
 * \brief Internal implmentation of simple random generator.
 */
	SimpleRNG RNG_;
/**
 * \brief The given point process we want to fit to the observed sequences of events.
 */
	IProcess* process_; 
/**
 * \brief The minimum tolerance of stopping the iterations of optimization.
 */
	double optTol;
/**
 * \brief Helper function of projected LBFGS.
 * @param[in] g     gradient of the parameters.
 * @param[in] s     helper variable for lbfgs update.
 * @param[in] y     helper variable for lbfgs update.
 * @param[in] Hdiag approximated Hessian matrix.
 * @param[out] d     updated parameters by the Hessian.
 */
	void lbfgs(const Eigen::VectorXd& g, const Eigen::MatrixXd& s, const Eigen::MatrixXd& y, const double& Hdiag, Eigen::VectorXd& d);

/**
 * \brief Helper function of updating lbfgs.
 * @param[in] y           helper variable for lbfgs update.
 * @param[in] s           helper variable for lbfgs update.
 * @param[in] corrections number of corrections of lbfgs update.
 * @param[in] old_dirs    the previous direction of updating the parameters.
 * @param[in] old_stps    the previous step size of updating the parameters.
 * @param[out] Hdiag      updated Hessian matrix.
 */
	void lbfgsUpdate(const Eigen::VectorXd& y, const Eigen::VectorXd& s, unsigned corrections, Eigen::MatrixXd& old_dirs, Eigen::MatrixXd& old_stps, double& Hdiag);

/**
 * \brief Identify the feasibility of the given solution.
 * @param  x current solution.
 * @return   whether the given solution is feasible or not.
 */
	bool isLegal(const Eigen::VectorXd& x);
/**
 * \brief Returns the components of the parameters needed to update.
 * @param[in] params  the current parameters.
 * @param[in] grad    the gradient of the current parameters.
 * @param[in] LB      lower bound of the parameters.
 * @param[in] UB      upper bound of the parameters.
 * @param[out] working index of the components needed to updated.
 */
	void ComputeWorkingSet(const Eigen::VectorXd& params, const Eigen::VectorXd& grad, double LB, double UB, Eigen::VectorXi& working);
/**
 * \brief Project the upated parameters into the box bounded by LB and UB.
 * @param[out] params the current parameters.
 * @param[in] LB     lower bound of the parameters.
 * @param[in] UB     upper bound of the parameters.
 */
	void projectBounds(Eigen::VectorXd& params, double LB, double UB);

/**
 * \brief the maximum number of iterations for updating the parameters.
 */
	unsigned maxIter_;

public:

/**
 * \brief The constructor.
 *
 * @param process the given point process model to fit.
 */
	Optimizer(IProcess* process) : process_(process)
	{
		RNG_.SetState(0, 0);
		optTol = 1e-10;
		maxIter_ = 1000;
	}

/**
 * \brief Stochastic gradient descend.
 * @param[in] gamma0       initial learning rate.
 * @param[in] ini_max_iter initial maximum number of iterations.
 * @param[in] data         the sampled sequence of events.
 */
	void SGD(const double& gamma0, const unsigned& ini_max_iter, const std::vector<Sequence>& data);
/**
 * \brief Projected LBFGS.
 * @param[in] LB lower bound of the parameters.
 * @param[in] UB upper bound of the parameters.
 */
	void PLBFGS(const double& LB, const double& UB);
/**
 * \brief Proximal method with group lasso type of regularization.
 * @param gamma0       initial learning rate.
 * @param lambda       regularization coefficient.
 * @param ini_max_iter initial maximum number of iterations.
 * @param group_size   the size of the entire parameter vector to be pushed into zero.
 */
	void ProximalGroupLasso(const double& gamma0, const double& lambda, const unsigned& ini_max_iter, const unsigned& group_size);
/**
 * \brief Proximal method with group lasso type of regularization for fitting Hawkes processes.
 * @param gamma0       initial learning rate.
 * @param lambda       regularization coefficient.
 * @param ini_max_iter initial maximum number of iterations.
 * @param group_size   the size of the entire parameter vector to be pushed into zero.
 */
	void ProximalGroupLassoForHawkes(const double& gamma0, const double& lambda, const unsigned& ini_max_iter, const unsigned& group_size);

/**
 * \brief Proximal method with nuclear norm regularization.
 * @param lambda         regularization coefficient.
 * @param rho            coefficient to enforce the low-rank constraint.
 * @param ini_max_iter   initial maximum number of iterations.
 * @param trueparameters true model parameters to compare with the current solution.
 */
	void ProximalNuclear(const double& lambda, const double& rho, const unsigned& ini_max_iter, const Eigen::VectorXd& trueparameters);

/**
 * \brief Proximal method using conditional gradient update.
 * @param gamma0         initial learning rate.
 * @param lambda         regularization coefficient.
 * @param rho            coefficient to enforce the low-rank constraint.
 * @param ub_alpha       upper bound of the nuclear norm of the excitation matrix.
 * @param ini_max_iter   initial maximum number of iterations.
 * @param trueparameters true model parameters to compare with the current solution.
 */
	void ProximalFrankWolfe(const double& gamma0, const double& lambda, const double& rho, const double& ub_alpha, const unsigned& ini_max_iter, const Eigen::VectorXd& trueparameters);
/**
 * \brief Proximal method using conditional gradient update.
 * @param gamma0         initial learning rate.
 * @param lambda         regularization coefficient.
 * @param rho            coefficient to enforce the low-rank constraint.
 * @param ub_alpha       upper bound of the nuclear norm of the excitation matrix.
 * @param ini_max_iter   initial maximum number of iterations.
 */
	void ProximalFrankWolfe(const double& gamma0, const double& lambda, const double& rho, const double& ub_alpha, const unsigned& ini_max_iter);
/**
 * \brief Proximal method using conditional gradient update for Low-rank Hawkes process.
 * @param gamma0       initial learning rate.
 * @param lambda0      regularization coefficient for the base intensity function.
 * @param lambda       regularization coefficient for the excitation matrix.
 * @param ub_lambda0   upper bound of the nuclear norm of the base intensity matrix.
 * @param ub_alpha     upper bound of the nuclear norm of the excitation matrix.
 * @param rho          coefficient to enforce the low-rank constraint.
 * @param ini_max_iter initial maximum number of iterations.
 * @param num_rows     the number of rows (or users) of the base intensity (or excitation) matrix.
 * @param num_cols     the number of cols (or items) of the base intensity (or excitation) matrix.
 * @param TrueLambda0  true model parameters of the base intensity matrix to compare with the current solution.
 * @param TrueAlpha    true model parameters of the excitation matrix to compare with the current solution.
 */
	void ProximalFrankWolfeForLowRankHawkes(const double& gamma0, const double& lambda0, const double& lambda, const double& ub_lambda0, const double& ub_alpha, const double& rho, const unsigned& ini_max_iter, const unsigned& num_rows, const unsigned& num_cols, const Eigen::MatrixXd& TrueLambda0, const Eigen::MatrixXd& TrueAlpha);
/**
 * \brief Proximal method using conditional gradient update for Low-rank Hawkes process.
 * @param gamma0       initial learning rate.
 * @param lambda0      regularization coefficient for the base intensity function.
 * @param lambda       regularization coefficient for the excitation matrix.
 * @param ub_lambda0   upper bound of the nuclear norm of the base intensity matrix.
 * @param ub_alpha     upper bound of the nuclear norm of the excitation matrix.
 * @param rho          coefficient to enforce the low-rank constraint.
 * @param ini_max_iter initial maximum number of iterations.
 * @param num_rows     the number of rows (or users) of the base intensity (or excitation) matrix.
 * @param num_cols     the number of cols (or items) of the base intensity (or excitation) matrix.
 */
	void ProximalFrankWolfeForLowRankHawkes(const double& gamma0, const double& lambda0, const double& lambda, const double& ub_lambda0, const double& ub_alpha, const double& rho, const unsigned& ini_max_iter, const unsigned& num_rows, const unsigned& num_cols);

};

#endif