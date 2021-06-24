#include <functional>
#include <thread>

namespace polyfem
{
	void par_for(const int size, const std::function<void(int, int, int)> &func);
	inline size_t get_n_threads() { return std::thread::hardware_concurrency(); }
}
