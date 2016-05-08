#include <iostream>
#include <iomanip> 
#include <chrono>
#include <Eigen/Dense>
#include "Sequence.h"
#include "PlainHawkes.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 1, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);
	params << 0.2, 0.8; 

	Eigen::MatrixXd beta(dim,dim);
	beta << 1;

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);
	std::vector<Sequence> sequences;

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	hawkes.Simulate(1000000, 1, sequences);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Simulating 1,000,000 events costs " << duration / 1000000.0 << " secs." << std::endl;
}
