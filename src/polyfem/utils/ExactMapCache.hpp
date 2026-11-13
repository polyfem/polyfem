#pragma once

#include <Eigen/Core>

#include <array>
#include <cstring>
#include <mutex>

namespace polyfem::utils
{
	/// @brief The last N results of a deterministic map of a vector, keyed by the bits of the input: a hit returns the
	/// stored result, which is what computing it again would give. Thread-safe; copies start empty.
	template <typename Out, int N = 4>
	class ExactMapCache
	{
	public:
		ExactMapCache() = default;
		ExactMapCache(const ExactMapCache &) {}
		ExactMapCache &operator=(const ExactMapCache &)
		{
			clear();
			return *this;
		}

		/// @brief The stored result for an input with the same bits, or compute() (stored, replacing the oldest result)
		template <typename F>
		Out get(const Eigen::VectorXd &in, const F &compute) const
		{
			{
				std::lock_guard<std::mutex> lock(mutex_);
				for (int i = 0; i < N; ++i)
					if (in_[i].size() > 0 && in_[i].size() == in.size()
						&& std::memcmp(in_[i].data(), in.data(), sizeof(double) * size_t(in.size())) == 0)
						return out_[i];
			}
			Out out = compute();
			std::lock_guard<std::mutex> lock(mutex_);
			in_[next_] = in;
			out_[next_] = out;
			next_ = (next_ + 1) % N;
			return out;
		}

		/// @brief Forget all results (the map changed)
		void clear()
		{
			std::lock_guard<std::mutex> lock(mutex_);
			for (auto &v : in_)
				v.resize(0);
			next_ = 0;
		}

	private:
		mutable std::mutex mutex_;
		mutable std::array<Eigen::VectorXd, N> in_;
		mutable std::array<Out, N> out_;
		mutable int next_ = 0;
	};
} // namespace polyfem::utils
