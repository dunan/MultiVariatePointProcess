/**
 * \file TriggeringKernel.h
 * \brief The class definition of TriggeringKernel which defines the general interface of a triggering kernel.
 */
#ifndef TRIGGERING_KERNEL_H
#define TRIGGERING_KERNEL_H

#include <Eigen/Dense>

/**
 * \class TriggeringKernel TriggeringKernel.h "include/TriggeringKernel.h"
 * \brief TriggeringKernel defines a general triggering kernel object of [Hawkes process](@ref PlainHawkes)
 */
class TriggeringKernel{

public:

/**
 * Get the value of the triggering kernel at time \f$t\f$.
 * @param[in]  t a given time
 * @return   the triggering kernel value at time \f$t\f$.
 */
	virtual double operator()(double t) = 0;

/**
 * Integral of the triggering kernel.
 * @param[in]  from start time of the integral.
 * @param[in]  to   end time of the integral.
 * @return      the value of the integral.
 */
	virtual double Integral(double from, double to) = 0;

/**
 * Upper bound of the triggering kernel in a given interval.
 * @param  from     start time of the interval.
 * @param  duration length of the interval.
 * @return          upper bound of the triggering kernel in the given interval.
 */
	virtual double Upper(double from, double duration) = 0;

};

#endif