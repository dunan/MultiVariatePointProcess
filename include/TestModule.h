#include "HPoisson.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"
#include "Diagnosis.h"
#include "Utility.h"
#include "PlainTerminating.h"
#include "TerminatingProcessLearningTriggeringKernel.h"
#include "Graph.h"

class TestModule
{

public:

	static void TestHPoisson();

	static void TestPlainHawkes();

	static void TestMultivariateTerminating();

	static void TestTerminatingProcessLearningTriggeringKernel();

};