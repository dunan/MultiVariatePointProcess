/**
 * \file PowerlawKernel.cc
 * \brief The class implementation of PowerlawKernel implementing the PowerLaw triggering kernel.
 */
#include <cmath>
#include <cassert>
#include <iostream>
#include "../include/PowerlawKernel.h"

double PowerlawKernel::operator()(double t)
{
	return (t >= sigma_ ? (beta_ / sigma_) * pow((t / sigma_), - beta_ - 1) : 0.0);
}

double PowerlawKernel::Integral(double from, double to)
{
	assert(from <= to);

	if(sigma_ < from)
	{
		return (pow(from / sigma_, -beta_) - pow(to / sigma_, -beta_));
	}else if(sigma_ <= to)
	{
		return (1 - pow(to / sigma_, - beta_));
	}

	return 0;
}

double PowerlawKernel::Upper(double from, double duration)
{
	double d = from + duration;
	if (((from <= sigma_) && (d > sigma_)) || (d <= sigma_))
	{
		return beta_ / sigma_;
	}else 
	{
		return (beta_ / sigma_) * pow((from / sigma_), - beta_ - 1);
	}
}