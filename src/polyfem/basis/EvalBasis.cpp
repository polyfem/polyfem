#include "polyfem/basis/BasisStore.hpp"
#include <polyfem/basis/EvalBasis.hpp>
#include <polyfem/basis/EvalLagrangeBasis.hpp>
#include <polyfem/basis/EvalRationalBasis.hpp>

#include <cassert>
#include <vector>

namespace polyfem::basis
{

	POLYFEM_BOTH int basis_count(const BasisDesc &desc)
	{
		if (desc.basis_family == BasisFamily::Lagrange)
		{
			return lagrange_basis_count(desc);
		}
		if (desc.basis_family == BasisFamily::Rational)
		{
			return rational_basis_count(desc);
		}
		if (desc.basis_family == BasisFamily::Unknown)
		{
			return desc.basis_num;
		}
		assert(false);
		return 0;
	}

	POLYFEM_BOTH void basis_values_single(
		int local_basis_index,
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values)
	{
		if (desc.basis_family == BasisFamily::Lagrange)
		{
			lagrange_basis_values_single(local_basis_index, desc, store, x, y, z, values);
			return;
		}
		if (desc.basis_family == BasisFamily::Rational)
		{
			rational_basis_values_single(local_basis_index, desc, store, x, y, z, values);
			return;
		}

		// Some basis does not support descriptor based eval. Use fallback callback.
#ifdef __CUDA_ARCH__
		assert(false && "Basis eval callback is not supported on device");
#else
		assert(desc.eval_callback_id >= 0 && desc.eval_callback_id < store.eval_callbacks.size());
		assert(local_basis_index >= 0 && local_basis_index < desc.basis_num);
		assert(values.size() == x.size());
		int expected_out_size = desc.basis_num * x.size();
		auto &callback = store.eval_callbacks[desc.eval_callback_id];

		std::vector<double> all_values(expected_out_size);
		std::vector<double> grad_x(expected_out_size);
		std::vector<double> grad_y(expected_out_size);
		std::vector<double> grad_z(expected_out_size);
		callback(x, y, z, all_values, grad_x, grad_y, grad_z);

		for (int i = 0; i < x.size(); ++i)
			values[i] = all_values[local_basis_index * x.size() + i];
#endif
	}

	POLYFEM_BOTH void basis_values(
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values)
	{

		if (desc.basis_family == BasisFamily::Lagrange)
		{
			lagrange_basis_values(desc, store, x, y, z, values);
			return;
		}
		if (desc.basis_family == BasisFamily::Rational)
		{
			rational_basis_values(desc, store, x, y, z, values);
			return;
		}

		// Some basis does not support descriptor based eval. Use fallback callback.
#ifdef __CUDA_ARCH__
		assert(false && "Basis eval callback is not supported on device");
#else
		assert(desc.eval_callback_id >= 0 && desc.eval_callback_id < store.eval_callbacks.size());
		int expected_out_size = desc.basis_num * x.size();
		assert(expected_out_size == values.size());
		auto &callback = store.eval_callbacks[desc.eval_callback_id];

		std::vector<double> grad_x(expected_out_size);
		std::vector<double> grad_y(expected_out_size);
		std::vector<double> grad_z(expected_out_size);
		callback(x, y, z, values, grad_x, grad_y, grad_z);
#endif
	}

	POLYFEM_BOTH void basis_gradients_single(
		int local_basis_index,
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		if (desc.basis_family == BasisFamily::Lagrange)
		{
			return lagrange_basis_gradients_single(local_basis_index, desc, store, x, y, z, grad_x, grad_y, grad_z);
		}
		else if (desc.basis_family == BasisFamily::Rational)
		{
			return rational_basis_gradients_single(local_basis_index, desc, store, x, y, z, grad_x, grad_y, grad_z);
		}

		// Some basis does not support descriptor based eval. Use fallback callback.
#ifdef __CUDA_ARCH__
		assert(false && "Basis eval callback is not supported on device");
#else
		assert(desc.eval_callback_id >= 0 && desc.eval_callback_id < store.eval_callbacks.size());
		assert(local_basis_index >= 0 && local_basis_index < desc.basis_num);
		assert(grad_x.size() == x.size());
		int expected_out_size = desc.basis_num * x.size();
		auto &callback = store.eval_callbacks[desc.eval_callback_id];

		std::vector<double> values(expected_out_size);
		std::vector<double> all_grad_x(expected_out_size);
		std::vector<double> all_grad_y(desc.dim > 1 ? expected_out_size : 0);
		std::vector<double> all_grad_z(desc.dim > 2 ? expected_out_size : 0);
		callback(x, y, z, values, all_grad_x, all_grad_y, all_grad_z);

		int offset = local_basis_index * x.size();
		for (int i = 0; i < x.size(); ++i)
		{
			grad_x[i] = all_grad_x[offset + i];
		}
		if (desc.dim > 1)
		{
			assert(grad_y.size() == x.size());
			for (int i = 0; i < x.size(); ++i)
			{
				grad_y[i] = all_grad_y[offset + i];
			}
		}
		if (desc.dim > 2)
		{
			assert(grad_z.size() == x.size());
			for (int i = 0; i < x.size(); ++i)
			{
				grad_z[i] = all_grad_z[offset + i];
			}
		}
#endif
	}

	POLYFEM_BOTH void basis_gradients(
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		if (desc.basis_family == BasisFamily::Lagrange)
		{
			return lagrange_basis_gradients(desc, store, x, y, z, grad_x, grad_y, grad_z);
		}
		else if (desc.basis_family == BasisFamily::Rational)
		{
			return rational_basis_gradients(desc, store, x, y, z, grad_x, grad_y, grad_z);
		}

		// Some basis does not support descriptor based eval. Use fallback callback.
#ifdef __CUDA_ARCH__
		assert(false && "Basis eval callback is not supported on device");
#else
		assert(desc.eval_callback_id >= 0 && desc.eval_callback_id < store.eval_callbacks.size());
		int expected_out_size = desc.basis_num * x.size();
		assert(expected_out_size == grad_x.size());
		auto &callback = store.eval_callbacks[desc.eval_callback_id];

		std::vector<double> values(expected_out_size);
		callback(x, y, z, values, grad_x, grad_y, grad_z);
#endif
	}

	POLYFEM_BOTH void basis_value_and_gradients(
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		if (desc.basis_family == BasisFamily::Lagrange)
		{
			lagrange_basis_value_and_gradients(desc, store, x, y, z, values, grad_x, grad_y, grad_z);
			return;
		}
		if (desc.basis_family == BasisFamily::Rational)
		{
			rational_basis_value_and_gradients(desc, store, x, y, z, values, grad_x, grad_y, grad_z);
			return;
		}

		// Some basis does not support descriptor based eval. Use fallback callback.
#ifdef __CUDA_ARCH__
		assert(false && "Basis eval callback is not supported on device");
#else
		assert(desc.eval_callback_id >= 0 && desc.eval_callback_id < store.eval_callbacks.size());
		int expected_out_size = desc.basis_num * x.size();
		assert(expected_out_size == grad_x.size());
		assert(expected_out_size == values.size());
		auto &callback = store.eval_callbacks[desc.eval_callback_id];

		callback(x, y, z, values, grad_x, grad_y, grad_z);
#endif
	}

} // namespace polyfem::basis
