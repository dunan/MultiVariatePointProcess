#include <cmath>
#include "../include/RayleighKernel.h"

double RayleighKernel::operator()(double t)
{
	double d = t - base_;

	return (d / (sigma_ * sigma_)) * exp(- (d * d) / (2 * sigma_ * sigma_));
}

double RayleighKernel::Integral(double from, double to)
{
	double a = from - base_;
	double b = to - base_;
	return (exp(- (a * a) / (2 * sigma_ * sigma_)) - exp(- (b * b) / (2 * sigma_ * sigma_)));
}

double RayleighKernel::Upper(double from, double duration)
{
	double d = from - base_ + duration;

	if ((from - base_ <= sigma_) && (d > sigma_))
	{
		return exp(-0.5) / sigma_;
	}else if (d <= sigma_)
	{
		return (d / (sigma_ * sigma_)) * exp(- (d * d) / (2 * sigma_ * sigma_));
	}else
	{
		d = from - base_;
		return (d / (sigma_ * sigma_)) * exp(- (d * d) / (2 * sigma_ * sigma_));
	}
}
