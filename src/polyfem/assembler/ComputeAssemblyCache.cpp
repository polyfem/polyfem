#include <polyfem/assembler/ComputeAssemblyCache.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/ComputeGeometryMapping.hpp>
#include <polyfem/basis/EvalBasis.hpp>

#include <cassert>
#include <Eigen/Core>

namespace polyfem::assembler
{
	namespace
	{
		template <int dim>
		Eigen::Vector<double, dim> xyz2vec(int id, Span<const double> x, Span<const double> y, Span<const double> z)
		{
			Eigen::Vector<double, dim> vec = Eigen::Vector<double, dim>::Zero();
			vec(0) = x[id];
			if constexpr (dim > 1)
			{
				vec(1) = y[id];
			}
			if constexpr (dim > 2)
			{
				vec(2) = z[id];
			}

			return vec;
		}

		template <int dim>
		void vec2xyz(int id, Eigen::Vector<double, dim> vec, Span<double> x, Span<double> y, Span<double> z)
		{
			x[id] = vec(0);
			if constexpr (dim > 1)
			{
				y[id] = vec(1);
			}
			if constexpr (dim > 2)
			{
				z[id] = vec(2);
			}
		}
	} // namespace

	template <int dim>
	void compute_assembly_cache_single(
		const AssemblyDataView &data,
		const AssemblyDataView &geom_data,
		int element_id,
		bool is_mass,
		AssemblyTempStorage &temp)
	{
		assert(0 <= element_id && element_id < data.element_desc.size());
		assert(0 <= element_id && element_id < geom_data.element_desc.size());

		auto &elem_desc = data.element_desc[element_id];
		auto &geom_elem_desc = geom_data.element_desc[element_id];
		auto &quad_desc = is_mass ? elem_desc.mass_quadrature_desc : elem_desc.quadrature_desc;

		temp.resize(dim, elem_desc.basis_desc.basis_num, geom_elem_desc.basis_desc.basis_num, quad_desc.w_range.num);

		// Quadrature points and weights.
		int quad_num = quad_desc.w_range.num;
		auto &quad_store = is_mass ? data.mass_quadrature_store : data.quadrature_store;
		auto quad_x = quad_store.get_x(quad_desc);
		auto quad_y = quad_store.get_y(quad_desc);
		auto quad_z = quad_store.get_z(quad_desc);
		auto quad_w = quad_store.get_w(quad_desc);

		// Compute geometry mapping.
		compute_geometry_mapping<dim>(
			geom_data,
			element_id,
			quad_x,
			quad_y,
			quad_z,
			temp.physical_x,
			temp.physical_y,
			temp.physical_z,
			temp.det_J,
			temp.J_inverse_transpose,
			temp.gbasis_values,
			temp.gbasis_grad_x,
			temp.gbasis_grad_y,
			temp.gbasis_grad_z);

		for (int qi = 0; qi < quad_num; ++qi)
			temp.weighted_measure[qi] = temp.det_J[qi] * quad_w[qi];

		// Compute basis values and gradients.
		int basis_num = elem_desc.basis_desc.basis_num;
		basis::basis_value_and_gradients(
			elem_desc.basis_desc,
			data.basis_store,
			quad_x,
			quad_y,
			quad_z,
			temp.basis_values,
			temp.basis_grad_x,
			temp.basis_grad_y,
			temp.basis_grad_z);

		// Compute basis grad physical.
		using Vec = Eigen::Vector<double, dim>;
		using Mat = Eigen::Matrix<double, dim, dim, Eigen::RowMajor>;
		for (int bi = 0; bi < basis_num; ++bi)
		{
			for (int qi = 0; qi < quad_num; ++qi)
			{
				Vec grad = xyz2vec<dim>(bi * quad_num + qi, temp.basis_grad_x, temp.basis_grad_y, temp.basis_grad_z);
				auto J_it = Eigen::Map<Mat>(temp.J_inverse_transpose.data() + dim * dim * qi);
				Vec grad_phy = J_it.transpose() * grad;
				vec2xyz<dim>(bi * quad_num + qi, grad_phy, temp.basis_grad_phy_x, temp.basis_grad_phy_y, temp.basis_grad_phy_z);
			}
		}
	}

	template void compute_assembly_cache_single<1>(const AssemblyDataView &, const AssemblyDataView &, int, bool, AssemblyTempStorage &);
	template void compute_assembly_cache_single<2>(const AssemblyDataView &, const AssemblyDataView &, int, bool, AssemblyTempStorage &);
	template void compute_assembly_cache_single<3>(const AssemblyDataView &, const AssemblyDataView &, int, bool, AssemblyTempStorage &);

	AssemblyCache compute_assembly_cache_batched(
		const AssemblyDataView &data,
		const AssemblyDataView &geom_data,
		bool is_mass)
	{
		AssemblyTempStorage temp;
		AssemblyCache cache;
		for (int e = 0; e < data.element_desc.size(); ++e)
		{
			// TODO: smart caching
			// - For low order basis, dont cache?
			// - Based on user json flag and element size.

			auto &elem_desc = data.element_desc[e];
			int dim = elem_desc.basis_desc.dim;

			switch (dim)
			{
			case 1:
				compute_assembly_cache_single<1>(data, geom_data, e, is_mass, temp);
				break;
			case 2:
				compute_assembly_cache_single<2>(data, geom_data, e, is_mass, temp);
				break;
			case 3:
				compute_assembly_cache_single<3>(data, geom_data, e, is_mass, temp);
				break;
			default:
				assert(false);
			}

			cache.append(is_mass, temp);
		}
		return cache;
	}

} // namespace polyfem::assembler
