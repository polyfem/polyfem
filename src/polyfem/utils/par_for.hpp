#pragma once

#include <functional>
#include <thread>

namespace polyfem
{
	namespace utils
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

		void par_for(const int size, const std::function<void(int, int, int)> &func);
		inline size_t get_n_threads() { return NThread::get().num_threads; }
	} // namespace utils
} // namespace polyfem
