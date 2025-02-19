#pragma once

#include <functional>
#include <thread>

#include <Eigen/Core>

#ifdef POLYFEM_WITH_TBB
#include <tbb/global_control.h>
#endif

namespace polyfem
{
	namespace utils
	{
		class NThread
		{
		public:
			static NThread &get()
			{
				static NThread instance;
				return instance;
			}

			inline size_t num_threads() const { return num_threads_; }

			void set_num_threads(const int max_threads)
			{
				const unsigned int tmp = max_threads <= 0 ? std::numeric_limits<int>::max() : max_threads;
				const unsigned int num_threads = std::min(tmp, std::thread::hardware_concurrency());

				num_threads_ = num_threads;
#ifdef POLYFEM_WITH_TBB
				thread_limiter = std::make_shared<tbb::global_control>(tbb::global_control::max_allowed_parallelism, num_threads);
#endif
				Eigen::setNbThreads(num_threads);
			}

		private:
			NThread() {}

			size_t num_threads_;

#ifdef POLYFEM_WITH_TBB
			/// limits the number of used threads
			std::shared_ptr<tbb::global_control> thread_limiter;
#endif
		};

		void par_for(const int size, const std::function<void(int, int, int)> &func);
		inline size_t get_n_threads() { return NThread::get().num_threads(); }
	} // namespace utils
} // namespace polyfem
