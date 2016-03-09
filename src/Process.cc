#include "../include/Process.h"

void IProcess::InitializeDimension(const std::vector<Sequence>& data)
{
	unsigned num_sequences_ = data.size();

	all_timestamp_per_dimension_ = std::vector<std::vector<std::vector<double> > >(num_sequences_, std::vector<std::vector<double> > (num_dims_, std::vector<double> ()));

	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();

		for(unsigned i = 0; i < seq.size(); ++ i)
		{
			all_timestamp_per_dimension_[c][seq[i].DimentionID].push_back(seq[i].time);
		}

	}
}