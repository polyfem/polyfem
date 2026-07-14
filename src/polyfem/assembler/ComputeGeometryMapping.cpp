#include <polyfem/assembler/ComputeGeometryMapping.hpp>

#include <polyfem/basis/EvalBasis.hpp>

#include <algorithm>
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
				vec(1) = y[id];
			if constexpr (dim > 2)
				vec(2) = z[id];
			return vec;
		}

		template <int dim>
		void vec2xyz(int id, const Eigen::Vector<double, dim> &vec, Span<double> x, Span<double> y, Span<double> z)
		{
			x[id] = vec(0);
			if constexpr (dim > 1)
				y[id] = vec(1);
			if constexpr (dim > 2)
				z[id] = vec(2);
		}

		// Build element local basis node position from dof mapping.
		template <int dim>
		Eigen::Vector<double, dim> basis_node_position(
			const AssemblyDataView &geom_data,
			const ElementDesc &geom_elem_desc,
			int local_basis_id)
		{
			int mapping_id = geom_elem_desc.dof_mapping_range.offset + local_basis_id;
			auto node_ids = geom_data.dof_mapping_store.get_node_ids(mapping_id);
			auto weights = geom_data.dof_mapping_store.get_weights(mapping_id);
			auto node_positions = geom_data.dof_mapping_store.get_positions(mapping_id);

			Eigen::Vector<double, dim> result = Eigen::Vector<double, dim>::Zero();
			for (int i = 0; i < node_ids.size(); ++i)
				result += weights[i] * Eigen::Map<const Eigen::Vector<double, dim>>(node_positions.data() + i * dim);
			return result;
		}

		template <int dim>
		void assert_position_sizes(
			int quad_num,
			Span<const double> y,
			Span<const double> z,
			Span<double> physical_x,
			Span<double> physical_y,
			Span<double> physical_z)
		{
			assert(physical_x.size() == quad_num);
			if constexpr (dim > 1)
			{
				assert(y.size() == quad_num);
				assert(physical_y.size() == quad_num);
			}
			else
			{
				assert(y.empty());
				assert(physical_y.empty());
			}

			if constexpr (dim > 2)
			{
				assert(z.size() == quad_num);
				assert(physical_z.size() == quad_num);
			}
			else
			{
				assert(z.empty());
				assert(physical_z.empty());
			}
		}
	} // namespace

	template <int dim>
	void compute_geometry_mapping(
		const AssemblyDataView &geom_data,
		int element_id,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> physical_x,
		Span<double> physical_y,
		Span<double> physical_z,
		Span<double> det_J,
		Span<double> J_inverse_transpose,
		Span<double> geom_basis_values,
		Span<double> geom_basis_grad_x,
		Span<double> geom_basis_grad_y,
		Span<double> geom_basis_grad_z)
	{
		assert(0 <= element_id && element_id < geom_data.element_desc.size());
		auto &geom_elem_desc = geom_data.element_desc[element_id];
		int quad_num = x.size();
		int geom_basis_num = geom_elem_desc.basis_desc.basis_num;

		assert_position_sizes<dim>(quad_num, y, z, physical_x, physical_y, physical_z);
		assert(det_J.size() == quad_num);
		assert(J_inverse_transpose.size() == quad_num * dim * dim);

		using Vec = Eigen::Vector<double, dim>;
		using Mat = Eigen::Matrix<double, dim, dim, Eigen::RowMajor>;

		// If basis already lives in physical space, no need to recompute position and jacobian.
		if (!geom_elem_desc.basis_desc.is_parametric)
		{
			std::copy(x.begin(), x.end(), physical_x.begin());
			if constexpr (dim > 1)
				std::copy(y.begin(), y.end(), physical_y.begin());
			if constexpr (dim > 2)
				std::copy(z.begin(), z.end(), physical_z.begin());

			for (int q = 0; q < quad_num; ++q)
			{
				Mat J_it = Mat::Identity();
				det_J[q] = 1.0;
				std::copy(J_it.data(), J_it.data() + J_it.size(), J_inverse_transpose.data() + dim * dim * q);
			}
			return;
		}

		assert(geom_basis_values.size() == geom_basis_num * quad_num);
		assert(geom_basis_grad_x.size() == geom_basis_num * quad_num);
		if constexpr (dim > 1)
			assert(geom_basis_grad_y.size() == geom_basis_num * quad_num);
		else
			assert(geom_basis_grad_y.empty());
		if constexpr (dim > 2)
			assert(geom_basis_grad_z.size() == geom_basis_num * quad_num);
		else
			assert(geom_basis_grad_z.empty());

		basis::basis_value_and_gradients(
			geom_elem_desc.basis_desc,
			geom_data.basis_store,
			x,
			y,
			z,
			geom_basis_values,
			geom_basis_grad_x,
			geom_basis_grad_y,
			geom_basis_grad_z);

		for (int q = 0; q < quad_num; ++q)
		{
			Vec phy_pos = Vec::Zero();
			Mat J = Mat::Zero();
			for (int b = 0; b < geom_basis_num; ++b)
			{
				// pos = sum(node pos * basis val)
				Vec basis_node_pos = basis_node_position<dim>(geom_data, geom_elem_desc, b);
				// J = basis grad * pos^T
				phy_pos += geom_basis_values[b * quad_num + q] * basis_node_pos;
				Vec grad = xyz2vec<dim>(b * quad_num + q, geom_basis_grad_x, geom_basis_grad_y, geom_basis_grad_z);
				J += grad * basis_node_pos.transpose();
			}

			// Compute J^-T and det(J).
			Mat J_it = J.inverse().transpose();
			vec2xyz<dim>(q, phy_pos, physical_x, physical_y, physical_z);
			det_J[q] = J.determinant();
			std::copy(J_it.data(), J_it.data() + J_it.size(), J_inverse_transpose.data() + dim * dim * q);
		}
	}

	template void compute_geometry_mapping<1>(const AssemblyDataView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>);
	template void compute_geometry_mapping<2>(const AssemblyDataView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>);
	template void compute_geometry_mapping<3>(const AssemblyDataView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>);

} // namespace polyfem::assembler
