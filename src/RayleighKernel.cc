/**
 * \file RayleighKernel.cc
 * \brief The class implementation of RayleighKernel implementing the Rayleigh triggering kernel.
 */
#include <cmath>
#include "../include/RayleighKernel.h"

double RayleighKernel::operator()(double t)
{
	return (t / (sigma_ * sigma_)) * exp(- (t * t) / (2 * sigma_ * sigma_));
}

double RayleighKernel::Integral(double from, double to)
{
	return (exp(- (from * from) / (2 * sigma_ * sigma_)) - exp(- (to * to) / (2 * sigma_ * sigma_)));
}

double RayleighKernel::Upper(double from, double duration)
{
	double d = from + duration;

	if ((from <= sigma_) && (d > sigma_))
	{
		return exp(-0.5) / sigma_;
	}else if (d <= sigma_)
	{
		return (d / (sigma_ * sigma_)) * exp(- (d * d) / (2 * sigma_ * sigma_));
	}else
	{
		d = from;
		return (d / (sigma_ * sigma_)) * exp(- (d * d) / (2 * sigma_ * sigma_));
	}
}
