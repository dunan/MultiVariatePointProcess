#include <vector>
#include <Event.h>

class Sequence
{

private:

	std::vector<Event> sequence_;

public:

	void Add(const Event& event) {sequence_.push_back(event);}

	unsigned Size() {return sequence_.size();}

};