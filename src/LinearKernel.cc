/**
 * \file LinearKernel.cc
 * \brief The class implementation of the linear triggering kernel.
 */
#include <cmath>
#include <cassert>
#include <iostream>
#include "../include/LinearKernel.h"

double LinearKernel::operator()(double t)
{
	return beta_ * t;
}

double LinearKernel::Integral(double from, double to)
{
	assert(from <= to);

	return 0.5 * beta_ * (to * to - from * from);
}

double LinearKernel::Upper(double from, double duration)
{
	return beta_ * (from + duration);
}