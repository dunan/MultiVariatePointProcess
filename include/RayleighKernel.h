/**
 * \file RayleighKernel.h
 * \brief The class definition of RayleighKernel implementing the Rayleigh triggering kernel.
 */
#ifndef RAYLEIGH_KERNEL_H
#define RAYLEIGH_KERNEL_H

#include "TriggeringKernel.h"

/**
 * \class RayleighKernel RayleighKernel.h "include/RayleighKernel.h"
 * \brief The Rayleigh triggering kernel.
 *
 * The Rayleigh triggering kernel is defined as: \f$\gamma(t,t_i) = \frac{t - t_i}{\sigma^2}e^{-(t - t_i)^2/(2\sigma^2)}\f$.
 */
class RayleighKernel : public TriggeringKernel
{

private:

	// expected time interval
	double sigma_;

public:

	RayleighKernel(double sigma) : sigma_(sigma){}

	virtual double operator()(double t);

	virtual double Integral(double from, double to);

	virtual double Upper(double from, double duration);

};

#endif