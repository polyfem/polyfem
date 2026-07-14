#pragma once

#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/Span.hpp>

#include <Eigen/Core>

#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CudaUtils.hpp>
#endif

namespace polyfem
{
	class DualVector
	{
	private:
		std::vector<double> host_values_;

#ifdef POLYFEM_WITH_CUDA
		DeviceBuf<double> device_values_;
#endif

	public:
		/// Build zero initialized dual vector of size.
		explicit DualVector(int size);

		int size() const { return host_values_.size(); }
		bool is_empty() const { return host_values_.empty(); }

		Span<double> host_view();
		Span<const double> host_view() const;

		/// Sum host and device view then return Eigen vector.
		Eigen::VectorXd to_eigen(ExecutionPolicy policy = {}) const;

#ifdef POLYFEM_WITH_CUDA
		/// Get device vector view. Lazily allocates device buffer.
		Span<double> device_view(ExecutionPolicy policy);
#endif
	};
} // namespace polyfem
