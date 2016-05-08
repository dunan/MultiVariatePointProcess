/**
 * \file SineKernel.h
 * \brief The class definition of SineKernel implementing the Sine triggering kernel.
 */
#ifndef SINE_KERNEL
#define SINE_KERNEL


#include "TriggeringKernel.h"

/**
 * \class SineKernel SineKernel.h "include/SineKernel.h"
 * \brief The Sine triggering kernel.
 *
 * The Sine triggering kernel is defined as: \f$\gamma(t,t_i) = \sin(t - t_i - 2\pi\lfloor\frac{t - t_i}{2\pi}\rfloor) + 1\f$.
 */
class SineKernel : public TriggeringKernel
{

private:

const double PI = 3.14159265358979323846;

public:

	virtual double operator()(double t);

	virtual double Integral(double from, double to);

	virtual double Upper(double from, double duration);

};

#endif