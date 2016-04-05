#ifndef RAYLEIGH_KERNEL_H
#define RAYLEIGH_KERNEL_H

#include "TriggeringKernel.h"

class RayleighKernel : public TriggeringKernel
{

private:

	// expected time interval
	double sigma_;

public:

	RayleighKernel(double& base, double& sigma) : TriggeringKernel(base), sigma_(sigma){}

	virtual double operator()(double t);

	virtual double Integral(double from, double to);

	virtual double Upper(double from, double duration);

};

#endif