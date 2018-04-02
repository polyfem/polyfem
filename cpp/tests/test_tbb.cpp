////////////////////////////////////////////////////////////////////////////////
#ifdef USE_TBB

#include <catch.hpp>
#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
////////////////////////////////////////////////////////////////////////////////

typedef tbb::enumerable_thread_specific< std::pair<int,int> > CounterType;
CounterType MyCounters (std::make_pair(0,0));

struct Body {
	void operator()(const tbb::blocked_range<int> &r) const {
		CounterType::reference my_counter = MyCounters.local();
		++my_counter.first;
		for (int i = r.begin(); i != r.end(); ++i)
			++my_counter.second;
	}
};

TEST_CASE("parallel_for", "[tbb]") {
    int n = tbb::task_scheduler_init::default_num_threads();
    tbb::task_scheduler_init init;
    REQUIRE(init.is_active());

	std::vector<int> data(100000);

	tbb::parallel_for( size_t(0), data.size(), [&]( size_t i ) {
        data[i] = -10;
    } );
}

TEST_CASE("parallel_for_memory", "[tbb]") {
	// tbb::parallel_for( tbb::blocked_range<int>(0, 100000000), Body());

	// for (CounterType::const_iterator i = MyCounters.begin(); i != MyCounters.end();  ++i)
	// {
	// 	printf("Thread stats:\n");
	// 	printf("  calls to operator(): %d", i->first);
	// 	printf("  total # of iterations executed: %d\n\n", i->second);
	// }
}

#endif
