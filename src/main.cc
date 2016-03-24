#include <iostream>
#include "../include/TestModule.h"

int main(int argc, char** argv)
{
	// Eigen::VectorXd x(10);
	// x << 10, 20, 30, 40, 50, 60, 70, 80, 90, 100;

	// Eigen::Map<Eigen::MatrixXd> y = Eigen::Map<Eigen::MatrixXd>(x.segment(1,4).data(),2,2);
	// std::cout << y << std::endl;

	// y(0,1) = -1;
	// std::cout << x << std::endl;

	// Eigen::Map<Eigen::VectorXd> z = Eigen::Map<Eigen::VectorXd>(x.segment(5,5).data(),5);
	// std::cout << z << std::endl;	
	// z(0) = -100;
	// std::cout << x << std::endl;	

	// Eigen::VectorXd x(4);
	// x << 10, 20, 30, 40;
	// std::cout << x << std::endl;
	// Eigen::Map<Eigen::MatrixXd> y = Eigen::Map<Eigen::MatrixXd>(x.data(),2,2);
	// std::cout << y << std::endl;
	// x(1) = 100;
	// std::cout << y << std::endl;
	// y(0,1) = -1;
	// std::cout << x << std::endl;
	// TestModule::TestHPoisson();

	// Eigen::VectorXd x(4);
	// x << 1,2,3,4;
	// std::cout << (x.array() < 2.5).cast<int>() << std::endl;
	
	// TestModule::TestPlainHawkes();

	// TestModule::TestMultivariateTerminating();

	// TestModule::TestTerminatingProcessLearningTriggeringKernel();

	// Eigen::MatrixXd A = (Eigen::MatrixXd::Random(10,10).array() + 1) / 2;
	// std::cout << A << std::endl;

	// RedSVD::RedSVD<Eigen::MatrixXd> svd;
	// svd.compute(A,2);
	// std::cout << svd.singularValues() << std::endl << std::endl;
	// std::cout << svd.matrixU() << std::endl << std::endl;
	// std::cout << svd.matrixV() << std::endl << std::endl;

	// Eigen::JacobiSVD<Eigen::MatrixXd> svdfull(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	// Eigen::MatrixXd S = svdfull.singularValues().asDiagonal();
	// std::cout <<  S << std::endl << std::endl;

	// std::cout << svdfull.matrixU() << std::endl << std::endl;
	// std::cout << svdfull.matrixV() << std::endl << std::endl;

	// std::cout << svdfull.matrixU() * S * svdfull.matrixV().transpose() << std::endl;

	// TestModule::TestPlainHawkesNuclear();

	// TestModule::TestLowRankHawkes();

	TestModule::TestGraph();

	return 0;
}