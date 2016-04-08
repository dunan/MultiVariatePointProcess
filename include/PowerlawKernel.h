#ifndef POWERLAW_KERNEL_H
#define POWERLAW_KERNEL_H

#include "TriggeringKernel.h"


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