#include <cmath>
#include "../include/ExpKernel.h"

double ExpKernel::operator()(double t)
{
	return exp(-beta_ * (t - base_));
}

double ExpKernel::Integral(double from, double to)
{
	return (exp(-beta_ * (from - base_)) - exp(-beta_ * (to - base_))) / beta_;
}

double ExpKernel::Upper(double from, double duration)
{
	return exp(-beta_ * (from - base_));
}