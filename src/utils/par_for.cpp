#include <polyfem/par_for.hpp>

#include <vector>

namespace polyfem
{
	void par_for(const int size, const std::function<void(int, int)> &func)
	{
#ifdef POLYFEM_WITH_CPP_THREADS
		const size_t n_threads = get_n_threads();
		std::vector<std::thread> threads(n_threads);

		for (int t = 0; t < n_threads; t++)
		{
			threads[t] = std::thread(std::bind(
				[&](int start, int end, int t) {
					par_for_thread_id = t; // Save the thread number in the global thread_local field
					func(start, end);
				},
				t * size / n_threads,
				(t + 1) == n_threads ? size : (t + 1) * size / n_threads,
				t));
		}
		std::for_each(threads.begin(), threads.end(), [](std::thread &x) { x.join(); });
#endif
	}
} // namespace polyfem
