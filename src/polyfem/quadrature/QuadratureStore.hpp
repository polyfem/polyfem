#pragma once

#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/CudaUtils.hpp>
#endif

namespace polyfem::quadrature
{

	struct QuadratureDesc
	{
		Range x_range;
		Range y_range;
		Range z_range;
		Range w_range;
	};

	struct QuadratureStoreView
	{
		Span<const double> x; //< Quadrature point x.
		Span<const double> y; //< Quadrature point y.
		Span<const double> z; //< Quadrature point z.
		Span<const double> w; //< Quadrature weight.

		POLYFEM_BOTH Span<const double> get_x(const QuadratureDesc &desc) const
		{
			return slice_by_range(x, desc.x_range);
		}

		POLYFEM_BOTH Span<const double> get_y(const QuadratureDesc &desc) const
		{
			return slice_by_range(y, desc.y_range);
		}

		POLYFEM_BOTH Span<const double> get_z(const QuadratureDesc &desc) const
		{
			return slice_by_range(z, desc.z_range);
		}

		POLYFEM_BOTH Span<const double> get_w(const QuadratureDesc &desc) const
		{
			return slice_by_range(w, desc.w_range);
		}

		Quadrature get_quadrature(const QuadratureDesc &desc) const;
	};

	class QuadratureStore
	{
	private:
		std::vector<double> x_; //< Quadrature point coordinate 0.
		std::vector<double> y_; //< Quadrature point coordinate 1.
		std::vector<double> z_; //< Quadrature point coordinate 2.
		std::vector<double> w_; //< Quadrature weight.

#ifdef POLYFEM_WITH_CUDA
		mutable bool need_host_device_sync_ = true;
		mutable DeviceBuf<double> d_x_;
		mutable DeviceBuf<double> d_y_;
		mutable DeviceBuf<double> d_z_;
		mutable DeviceBuf<double> d_w_;
#endif

	public:
		QuadratureStoreView view() const;

		/// Append quadrature to the store. Require non-empty quadrature.
		QuadratureDesc append(const Quadrature &quad);

#ifdef POLYFEM_WITH_CUDA
		/// Return view on device. Lazily sync data.
		QuadratureStoreView device_view(ExecutionPolicy policy) const;

		/// Release device storage.
		void clear_device_storage();
#endif
	};
} // namespace polyfem::quadrature
