////////////////////////////////////////////////////////////////////////////////
#ifdef POLYFEM_WITH_TBB

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("parallel_for", "[tbb_test]")
{
	std::vector<int> data(100000);

	tbb::parallel_for(size_t(0), data.size(), [&](size_t i) { data[i] = -10; });
}

TEST_CASE("parallel_for_memory", "[tbb_test]")
{
	typedef tbb::enumerable_thread_specific<std::pair<int, int>> CounterType;
	CounterType counters(std::make_pair(0, 0));

	tbb::parallel_for(tbb::blocked_range<int>(0, 100000000), [&](const tbb::blocked_range<int> &r) {
		CounterType::reference loc_counter = counters.local();
		++loc_counter.first;
		for (int i = r.begin(); i != r.end(); ++i)
			++loc_counter.second;
	});

	for (CounterType::const_iterator i = counters.begin(); i != counters.end(); ++i)
	{
		printf("Thread stats:\n");
		printf("  calls to operator(): %d", i->first);
		printf("  total # of iterations executed: %d\n\n", i->second);
	}
}

#endif
