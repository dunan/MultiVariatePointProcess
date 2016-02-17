#include <vector>
#include <Process.h>
#include <Sequence.h>

class PoissonMLELearner : public Learner
{

private:

	PoissonProcess poisson_;

public:

	PoissonMLELearner(const unsigned& num_dims_){}
	~PoissonProcess(){}
};