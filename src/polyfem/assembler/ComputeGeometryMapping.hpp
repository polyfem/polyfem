#pragma once

#include <polyfem/assembler/AssemblyData.hpp>

namespace polyfem::assembler
{
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
		Span<double> geom_basis_grad_z);

} // namespace polyfem::assembler
