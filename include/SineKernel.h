#ifndef SINE_KERNEL
#define SINE_KERNEL


#include "TriggeringKernel.h"


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