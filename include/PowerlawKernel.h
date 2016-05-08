/**
 * \file PowerlawKernel.h
 * \brief The class definition of PowerlawKernel implementing the PowerLaw triggering kernel.
 */
#ifndef POWERLAW_KERNEL_H
#define POWERLAW_KERNEL_H

#include "TriggeringKernel.h"

/**
 * \class PowerlawKernel PowerlawKernel.h "include/PowerlawKernel.h"
 * \brief The Power-Law triggering kernel.
 *
 * The Power-Law triggering kernel is defined as: \f$\gamma(t,t_i) = \frac{\beta}{\sigma}\bigg(\frac{t-t_i}{\sigma}\bigg)^{-\beta-1}\f$, \f$t - t_i\geq\sigma\f$.
 */
class PowerlawKernel : public TriggeringKernel
{

private:

	// expected time interval
	double beta_;
	double sigma_;

public:

	PowerlawKernel(double beta, double sigma) : beta_(beta), sigma_(sigma){}

	virtual double operator()(double t);

	virtual double Integral(double from, double to);

	virtual double Upper(double from, double duration);

};

#endif