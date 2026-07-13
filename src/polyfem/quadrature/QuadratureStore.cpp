#include <polyfem/quadrature/QuadratureStore.hpp>

#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/quadrature/Quadrature.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <cassert>
#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CudaUtils.hpp>
#endif

namespace polyfem::quadrature
{

	QuadratureStoreView QuadratureStore::view() const
	{
		return QuadratureStoreView{x_, y_, z_, w_};
	}

	Quadrature QuadratureStoreView::get_quadrature(const QuadratureDesc &desc) const
	{
		auto x_span = get_x(desc);
		auto y_span = get_y(desc);
		auto z_span = get_z(desc);
		auto w_span = get_w(desc);
		int num = w_span.size();

		int dim = 3;
		if (z_span.empty())
			dim = 2;
		if (y_span.empty())
			dim = 1;
		Quadrature quad;

		quad.points.resize(num, dim);
		quad.weights.resize(num);
		for (int i = 0; i < num; ++i)
		{
			quad.points(i, 0) = x_span[i];
			if (dim > 1)
				quad.points(i, 1) = y_span[i];
			if (dim > 2)
				quad.points(i, 2) = z_span[i];
			quad.weights(i) = w_span[i];
		}
		return quad;
	}

	/// @brief Append quadrature to the store. Require non-empty quadrature.
	QuadratureDesc QuadratureStore::append(const Quadrature &quad)
	{
		assert(quad.size() != 0);

#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		// Quadrature class stores:
		// - points: A matrix of size (quad_num x dim). Each row is a quadrature point.
		// - weights: A vector of size quad_num.

		QuadratureDesc desc;
		int dim = quad.points.cols();
		// dim == 1, y and z are empty.
		if (dim == 1)
		{
			auto col_x = quad.points.col(0);
			desc.x_range = Range{static_cast<int>(x_.size()), static_cast<int>(col_x.size())};
			desc.y_range = {};
			desc.z_range = {};
			desc.w_range = Range{static_cast<int>(w_.size()), static_cast<int>(quad.weights.size())};

			x_.insert(x_.end(), col_x.begin(), col_x.end());
			w_.insert(w_.end(), quad.weights.begin(), quad.weights.end());
		}
		// dim == 2, z is empty.
		else if (dim == 2)
		{
			auto col_x = quad.points.col(0);
			auto col_y = quad.points.col(1);
			desc.x_range = Range{static_cast<int>(x_.size()), static_cast<int>(col_x.size())};
			desc.y_range = Range{static_cast<int>(y_.size()), static_cast<int>(col_y.size())};
			desc.z_range = {};
			desc.w_range = Range{static_cast<int>(w_.size()), static_cast<int>(quad.weights.size())};

			x_.insert(x_.end(), col_x.begin(), col_x.end());
			y_.insert(y_.end(), col_y.begin(), col_y.end());
			w_.insert(w_.end(), quad.weights.begin(), quad.weights.end());
		}
		else if (dim == 3)
		{
			auto col_x = quad.points.col(0);
			auto col_y = quad.points.col(1);
			auto col_z = quad.points.col(2);
			desc.x_range = Range{static_cast<int>(x_.size()), static_cast<int>(col_x.size())};
			desc.y_range = Range{static_cast<int>(y_.size()), static_cast<int>(col_y.size())};
			desc.z_range = Range{static_cast<int>(z_.size()), static_cast<int>(col_z.size())};
			desc.w_range = Range{static_cast<int>(w_.size()), static_cast<int>(quad.weights.size())};

			x_.insert(x_.end(), col_x.begin(), col_x.end());
			y_.insert(y_.end(), col_y.begin(), col_y.end());
			z_.insert(z_.end(), col_z.begin(), col_z.end());
			w_.insert(w_.end(), quad.weights.begin(), quad.weights.end());
		}
		else
		{
			assert(false && "Invalid dimension");
		}

		return desc;
	}

#ifdef POLYFEM_WITH_CUDA
	QuadratureStoreView QuadratureStore::device_view(ExecutionPolicy policy) const
	{
		if (need_host_device_sync_)
		{
			assert(policy.stream && policy.mr);
			d_x_ = copy_to_device_async<double>(x_, policy);
			d_y_ = copy_to_device_async<double>(y_, policy);
			d_z_ = copy_to_device_async<double>(z_, policy);
			d_w_ = copy_to_device_async<double>(w_, policy);
			policy.stream->sync();
			need_host_device_sync_ = false;
		}
		return QuadratureStoreView{*d_x_, *d_y_, *d_z_, *d_w_};
	}

	void QuadratureStore::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_x_ = {};
		d_y_ = {};
		d_z_ = {};
		d_w_ = {};
	}
#endif

} // namespace polyfem::quadrature
