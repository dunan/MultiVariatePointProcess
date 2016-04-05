#ifndef EXP_KERNEL_H
#define EXP_KERNEL_H

#include "TriggeringKernel.h"


class ExpKernel : public TriggeringKernel
{

private:

	// expected time interval
	double beta_;

public:

	ExpKernel(double base, double beta) : TriggeringKernel(base), beta_(beta){}

	virtual double operator()(double t);

	virtual double Integral(double from, double to);

	virtual double Upper(double from, double duration);

};

#endif