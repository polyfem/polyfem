#pragma once

#include <functional>
#include <thread>

namespace polyfem
{
	class NThread
	{
	public:
		size_t num_threads;
		static NThread &get()
		{
			static NThread instance;
			return instance;
		}

	private:
		NThread() {}
	};

	thread_local static int par_for_thread_id = -1; // This will be set inside par_for
	void par_for(const int size, const std::function<void(int, int)> &func);
	inline size_t get_n_threads() { return NThread::get().num_threads; }
} // namespace polyfem
