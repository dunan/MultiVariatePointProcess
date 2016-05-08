/**
 * \file ExpKernel.cc
 * \brief The class implementation of ExpKernel implementing the Exponential triggering kernel.
 */
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