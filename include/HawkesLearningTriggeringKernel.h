#ifndef HAWKES_LEARNING_TRIGGERING_KERNEL_H
#define HAWKES_LEARNING_TRIGGERING_KERNEL_H

#include <vector>
#include <cmath>
#include <string>
#include <map>
#include "Process.h"
#include "Optimizer.h"
#include "Graph.h"

class HawkesLearningTriggeringKernel : public IProcess
{

public:

	enum Regularizer {L22, GROUP, NONE};
	enum RegCoef {LAMBDA0, LAMBDA};

	//  Records the options

	struct OPTION
	{
		Regularizer base_intensity_regularizer;
		Regularizer excitation_regularizer;
		std::map<RegCoef, double> coefficients;	
	};

protected:

	const double PI = 3.14159265358979323846;

//  Basis of the triggering kernel
	Eigen::VectorXd tau_;
	
//	Number of RBF
	unsigned num_rbfs_;

//	Kernel bandwidth
	Eigen::VectorXd sigma_;
	Eigen::VectorXd sqrt2sigma_;
	Eigen::VectorXd sqrt2PIsigma_;
	Eigen::VectorXd erfctau_sigma_;	

// Temporal features associated with the intensity function
	std::vector<std::vector<std::vector<Eigen::MatrixXd> > > arrayK;

// Temporal features derived from the integral of the intensity 
	std::vector<std::vector<Eigen::MatrixXd> > arrayG;

	Eigen::VectorXd observation_window_T_;

//  This variable is process-specific. It records totoal number of sequences used to fit the process.
	unsigned num_sequences_;

//  This function requires process-specific implementation. It initializes the temporal features used to calculate the negative loglikelihood and the gradient. 
	void Initialize(const std::vector<Sequence>& data);

//  This function requires process-specific implementation. It initializes the temporal features used to calculate the negative loglikelihood and the gradient when the network structure is known.
	void InitializeWithGraph(const std::vector<Sequence>& data);

	void InitializeConstants();

	void PostProcessing();

	void GetNegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	OPTION options_;

	const Graph* graph_;

public:

	//  Constructor : n is the number of parameters in total; num_dims is the number of dimensions in the process;
	HawkesLearningTriggeringKernel(const unsigned& n, const unsigned& num_dims, const Eigen::VectorXd& tau, const Eigen::VectorXd& sigma) : IProcess(n, num_dims), tau_(tau), sigma_(sigma), num_sequences_(0), graph_(NULL)
	{
		HawkesLearningTriggeringKernel::InitializeConstants();
	}

//  Constructor : n is the number of parameters in total; num_dims is the number of dimensions in the process; graph is the pointer to the given network structure;
	HawkesLearningTriggeringKernel(const unsigned& n, const unsigned& num_dims, const Graph* graph, const Eigen::VectorXd& tau, const Eigen::VectorXd& sigma) : IProcess(n, num_dims), tau_(tau), sigma_(sigma), num_sequences_(0), graph_(graph)
	{
		HawkesLearningTriggeringKernel::InitializeConstants();
	}

	//  MLE esitmation of the parameters
	void fit(const std::vector<Sequence>& data, const OPTION& options);

//  This virtual function requires process-specific implementation. It calculates the negative loglikelihood of the given data. This function must be called after the Initialize method to return the negative loglikelihood of the data with respect to the current parameters. 
//	The returned negative loglikelihood is stored in the variable objvalue;
//	The returned gradient vector wrt the current parameters is stored in the variable Gradient; 
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	//  Return the stochastic gradient on the random sample k.
	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
	virtual double IntensityUpperBound(const double& t, const double& L, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the integral of the intensity from a to b
	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

//  This function predicts the next event by simulation;
	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

	void PlotTriggeringKernel(const unsigned& dim_m, const unsigned& dim_n, const double& T, const double& delta);

};


#endif