#ifndef LINEAR_KERNEL
#define LINEAR_KERNEL


#include "TriggeringKernel.h"


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