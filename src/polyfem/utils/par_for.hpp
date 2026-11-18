#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <thread>

#include <Eigen/Core>

#if defined(POLYFEM_WITH_TBB) || defined(POLYFEM_WITH_CPP_THREADS)
#include <tbb/global_control.h>
#endif

namespace polyfem
{
	namespace utils
	{
		/// Restrict parallel work on the current thread and its CPP worker calls.
		class ScopedThreadLimit
		{
		public:
			explicit ScopedThreadLimit(const size_t limit) : previous_(limit_)
			{
				limit_ = std::min(limit_, std::max(size_t(1), limit));
			}
			~ScopedThreadLimit() { limit_ = previous_; }
			ScopedThreadLimit(const ScopedThreadLimit &) = delete;
			ScopedThreadLimit &operator=(const ScopedThreadLimit &) = delete;
			static size_t current() { return limit_; }

		private:
			inline static thread_local size_t limit_ = std::numeric_limits<size_t>::max();
			size_t previous_;
		};

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
#if defined(POLYFEM_WITH_TBB) || defined(POLYFEM_WITH_CPP_THREADS)
				thread_limiter = std::make_shared<tbb::global_control>(tbb::global_control::max_allowed_parallelism, num_threads);
#endif
				Eigen::setNbThreads(num_threads);
			}

		private:
			NThread() {}

			size_t num_threads_ = 1;

#if defined(POLYFEM_WITH_TBB) || defined(POLYFEM_WITH_CPP_THREADS)
			/// limits the number of used threads
			std::shared_ptr<tbb::global_control> thread_limiter;
#endif
		};

		void par_for(const int size, const std::function<void(int, int, int)> &func);
		inline size_t get_n_threads() { return std::min(NThread::get().num_threads(), ScopedThreadLimit::current()); }
	} // namespace utils
} // namespace polyfem
