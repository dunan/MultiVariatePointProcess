/**
 * \file LinearKernel.h
 * \brief The class definition of the linear triggering kernel.
 */
#ifndef LINEAR_KERNEL_H
#define LINEAR_KERNEL_H


#include "TriggeringKernel.h"

/**
 * \class LinearKernel LinearKernel.h "include/LinearKernel.h"
 * \brief The linear triggering kernel is defined as: \f$\gamma(t,t_i) = \beta(t - t_i)\f,\beta\geq 0$. 
 */
class LinearKernel : public TriggeringKernel
{

private:

	// expected time interval
	double beta_;

public:

	LinearKernel(double beta) : beta_(beta){}

	virtual double operator()(double t);

	virtual double Integral(double from, double to);

	virtual double Upper(double from, double duration);

};

#endif