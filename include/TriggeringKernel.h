#ifndef TRIGGERING_KERNEL_H
#define TRIGGERING_KERNEL_H

#include <Eigen/Dense>


class TriggeringKernel{

public:

	virtual double operator()(double t) = 0;

	virtual double Integral(double from, double to) = 0;

	virtual double Upper(double from, double duration) = 0;

};

#endif