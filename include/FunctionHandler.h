#ifndef FUNCTION_HANDLER_H
#define FUNCTION_HANDLER_H
#include <Eigen/Dense>

class FunctionHandler
{

public:

/**
 * Get the value of the functor at time \f$t\f$.
 * @param[in]	t	a column vector of given time
 * @param[out]	y	the function value at time \f$t\f$.
 */
	virtual void operator()(const Eigen::VectorXd& t, Eigen::VectorXd& y) = 0;
};

#endif