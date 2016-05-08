/**
 * \file SineKernel.cc
 * \brief The class implementation of SineKernel implementing the Sine triggering kernel.
 */
#include <cmath>
#include <cassert>
#include <iostream>
#include "../include/SineKernel.h"

double SineKernel::operator()(double t)
{
	unsigned p = t / (2 * PI);

	double t_hat = t - p * (2 * PI);

	return sin(t_hat) + 1;
}

double SineKernel::Integral(double from, double to)
{
	assert(from <= to);

	return (to - from + cos(from) - cos(to));
}

double SineKernel::Upper(double from, double duration)
{
	unsigned p = from / (2 * PI);
	double t = from - p * (2 * PI);

	p = duration / (2 * PI);
	double d = duration - p * (2 * PI);

	if(t + d <= 0.5 * PI)
	{
		return sin(t + d) + 1;
	}else if (t <= 0.5 * PI)
	{
		return 2;
	}else
	{
		return fmax(sin(t), sin(t + d)) + 1.0;
	}
}