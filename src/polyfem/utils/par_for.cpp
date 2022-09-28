#include "par_for.hpp"

#include <vector>
#include <algorithm>

namespace polyfem
{
	namespace utils
	{
		void par_for(const int size, const std::function<void(int, int, int)> &func)
		{
#ifdef POLYFEM_WITH_CPP_THREADS
			const size_t n_threads = get_n_threads();
			std::vector<std::thread> threads(n_threads);

			for (int t = 0; t < n_threads; t++)
			{
				threads[t] = std::thread(std::bind(
					func,
					t * size / n_threads,
					(t + 1) == n_threads ? size : (t + 1) * size / n_threads,
					t));
			}
			std::for_each(threads.begin(), threads.end(), [](std::thread &x) { x.join(); });
#endif
		}
	} // namespace utils
} // namespace polyfem
