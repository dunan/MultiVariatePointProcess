#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "LowRankHawkesProcess.h"

int main(const int argc, const char** argv)
{
	unsigned num_users = 64, num_items = 64;
	std::vector<Sequence> data;
	std::cout << "1. Loading " << num_users << " users " << num_items << " items " << " with 1000 events each" << std::endl;
	ImportFromExistingUserItemSequences("data/low_rank_hawkes_sampled_entries_events", num_users, num_items, data);
	unsigned dim = num_users * num_items;
	Eigen::VectorXd beta = Eigen::VectorXd::Constant(dim, 1.0);
	LowRankHawkesProcess low_rank_hawkes(num_users, num_items, beta);
	LowRankHawkesProcess::OPTION options;
	options.coefficients[LowRankHawkesProcess::LAMBDA0] = 0;
	options.coefficients[LowRankHawkesProcess::LAMBDA] = 0;

	Eigen::MatrixXd TrueLambda0, TrueAlpha;
	LoadEigenMatrixFromTxt("data/truth-syn-Lambda0", num_users, num_items, TrueLambda0);
	LoadEigenMatrixFromTxt("data/truth-syn-Alpha", num_users, num_items, TrueAlpha);
	std::cout << "2. Fitting Parameters " << std::endl;
	low_rank_hawkes.fit(data, options, TrueLambda0, TrueAlpha);

	return 0;
}