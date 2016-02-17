#include <vector>
#include <Process.h>
#include <Sequence.h>

class HawkesMLELearner : public Learner
{

private:

	HawkesProcess hawkes_;

public:

	HawkesMLELearner(const unsigned& num_dims_){}
	~HawkesMLELearner(){}
};