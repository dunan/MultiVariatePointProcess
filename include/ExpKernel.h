/**
 * \file ExpKernel.h
 * \brief The class definition of ExpKernel implementing the Exponential triggering kernel.
 */
#ifndef EXP_KERNEL_H
#define EXP_KERNEL_H

#include "TriggeringKernel.h"

/**
 * \class ExpKernel ExpKernel.h "include/ExpKernel.h"
 * \brief The Exponential triggering kernel.
 *
 * The Exponential triggering kernel is defined as: \f$\gamma(t,t_i) = \exp(-\beta(t - t_i))\f$.
 */
class ExpKernel : public TriggeringKernel
{

private:

	// expected time interval
	double beta_;

public:

	ExpKernel(double beta) : beta_(beta){}

	virtual double operator()(double t);

	virtual double Integral(double from, double to);

	virtual double Upper(double from, double duration);

};

#endif