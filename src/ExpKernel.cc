#include <cmath>
#include "../include/ExpKernel.h"

double ExpKernel::operator()(double t)
{
	return exp(-beta_ * t);
}

double ExpKernel::Integral(double from, double to)
{
	return (exp(-beta_ * (from)) - exp(-beta_ * (to))) / beta_;
}

double ExpKernel::Upper(double from, double duration)
{
	return exp(-beta_ * (from));
}