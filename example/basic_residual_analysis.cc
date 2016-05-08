#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "HawkesGeneralKernel.h"
#include "Diagnosis.h"
#include "SineKernel.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 1, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);
	params << 0.5, 0.1;

	std::vector<std::vector<TriggeringKernel*> > triggeringkernels(dim, std::vector<TriggeringKernel*>(dim, NULL));
	for(unsigned m = 0; m < dim; ++ m)
	{
		for(unsigned n = 0; n < dim; ++ n)
		{
			triggeringkernels[m][n] = new SineKernel(); 
		}
	}

	HawkesGeneralKernel hawkes(num_params, dim, triggeringkernels);
	hawkes.SetParameters(params);

	OgataThinning ot(dim);
	std::vector<Sequence> sequences;
	ot.Simulate(hawkes, 50, 1, sequences);
	
	std::cout << "Residual analysis fitting: " << Diagnosis::TimeChangeFit(hawkes, sequences[0]) << std::endl;

	hawkes.PlotIntensityFunction(sequences[0]);
	return 0;
}