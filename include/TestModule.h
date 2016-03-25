#include "HPoisson.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"
#include "Diagnosis.h"
#include "Utility.h"
#include "PlainTerminating.h"
#include "TerminatingProcessLearningTriggeringKernel.h"
#include "Graph.h"
#include "LowRankHawkesProcess.h"
#include "ContinEst.h"

class TestModule
{

public:

	static void TestHPoisson();

	static void TestPlainHawkes();

	static void TestMultivariateTerminating();

	static void TestTerminatingProcessLearningTriggeringKernel();

	static void TestPlainHawkesNuclear();

	static void TestLowRankHawkes();

	static void TestGraph();

	static void TestPlot();

};