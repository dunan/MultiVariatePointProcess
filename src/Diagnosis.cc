#include "../include/Diagnosis.h"
#include <iostream>

double Diagnosis::TimeChangeFit(IProcess& process, const Sequence& seq)
{
	const std::vector<Event>& events = seq.GetEvents();

	Eigen::VectorXd samples = Eigen::VectorXd::Zero(events.size());

	samples(0) = process.IntensityIntegral(0, events[0].time, seq);
	for(unsigned i = 0; i < events.size() - 1; ++ i)
	{
		samples(i + 1) = process.IntensityIntegral(events[i].time, events[i + 1].time, seq);
	}

	std::cout << samples.transpose() << std::endl;

	return samples.array().sum() / samples.size();
}