/**
 * \file SimpleRNG.h
 * \brief The class definition of SimpleRNG implementing a simple random number generator.
 */
#ifndef SIMPLERNG_H
#define SIMPLERNG_H

/**
 * \class SimpleRNG SimpleRNG.h "include/SimpleRNG.h"
 * \brief A simple C++ random number generator from John D. Cook.
 */
class SimpleRNG
{
public:
    
    SimpleRNG();

    /**
     * \brief Seed the random number generator. 
     * @param[in] u random seed
     * @param[in] v random seed
     */
    void SetState(unsigned int u, unsigned int v);

    /**
     * \brief Extract the internal state of the generator
     * @param[out] u internal seed
     * @param[out] v internal seed
     */
    void GetState(unsigned int& u, unsigned int& v);

    /**
     * A uniform random sample from the open interval (0, 1) 
     * @return a uniform random sample from the open interval (0, 1) 
     */
    double GetUniform();
    /**
     * A uniform random sample from the set of unsigned integers 
     * @return a uniform random sample from the set of unsigned integers 
     */
    unsigned int GetUint();
    /**
     * Get a uniform random value.
     *
     * This stateless version makes it more convenient to get a uniform random value and transfer the state in and out in one operation.
     * @param[out]  u internal state
     * @param[out]  v internal state
     * @return   a uniform sample from (0, 1)
     */
    double GetUniform(unsigned int& u, unsigned int& v); 
    /**
     * This stateless version makes it more convenient to get a random unsigned integer and transfer the state in and out in one operation. 
     * @param  u internal state
     * @param  v internal state
     * @return   a uniform integer sample
     */
    unsigned int GetUint(unsigned int& u, unsigned int& v);
    /**
     * Normal (Gaussian) random sample.
     * @param  mean              the expectation of Gaussian
     * @param  standardDeviation the standard deviation of Gaussian
     * @return                   a sample from Gaussian
     */
    double GetNormal(double mean, double standardDeviation);
    /**
     * Exponential random sample 
     * @param  mean the expectation of exponential distribution
     * @return      a sample from the exponential distribution
     */
    double GetExponential(double mean);

    /**
     * Gamma random sample
     * @param  shape the shape parameter
     * @param  scale the scale parameter
     * @return       a sample from Gamma distribution
     */
    double GetGamma(double shape, double scale);

    /**
     * Chi-square sample
     * @param  degreesOfFreedom parameter of the ChiSquare distribution
     * @return                  a sample from ChiSquare distribution
     */
    double GetChiSquare(double degreesOfFreedom);

	
    /**
     * Inverse-gamma sample
     * @param  shape the shape parameter
     * @param  scale the scale parameter
     * @return       a sample from Inverse-gamma distribution
     */
    double GetInverseGamma(double shape, double scale);

	/**
     * Weibull sample
     * @param  shape the shape parameter
     * @param  scale the scale parameter
     * @return       a sample from the Weibull distribution
     */
    double GetWeibull(double shape, double scale);

    /**
     * Cauchy sample
     * @param  median the median parameter
     * @param  scale the scale parameter
     * @return       a sample from the Cauchy distribution
     */
    double GetCauchy(double median, double scale);
    /**
     * Student-t sample
     * @param  degreesOfFreedom degree of freedom
     * @return                  a sample from the Student-t distribution
     */
    double GetStudentT(double degreesOfFreedom);

    /**
     * The Laplace distribution is also known as the double exponential distribution.
     * @param  mean  the expectation of the distribution
     * @param  scale the scale parameter
     * @return       a sample from Laplace distribution
     */
    double GetLaplace(double mean, double scale);

    /**
     * Log-normal sample
     * @param  mu    the location parameter 
     * @param  sigma the scale parameter
     * @return       a sample from Log-normal distribution
     */
    double GetLogNormal(double mu, double sigma);

    /**
     * Beta sample
     * @param  a the shape parameter
     * @param  b the shape parameter
     * @return   a sample from Beta distribution
     */
    double GetBeta(double a, double b);
    /**
     * Poisson sample
     * @param  lambda expectation of Poisson distribution
     * @return        a sample from Poisson distribution
     */
	int GetPoisson(double lambda);

private:
    unsigned int m_u, m_v;
	int PoissonLarge(double lambda);
	int PoissonSmall(double lambda);
	double LogFactorial(int n);
};


#endif