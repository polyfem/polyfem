#pragma once

#include <vector>
#include <Eigen/Core>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

namespace polyfem::solver
{
	class InertiaForceDerivative
	{
	public:
		static void force_shape_derivative(
			const InertiaForm &form,
			bool is_volume,
			const int n_geom_bases,
			const double t,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::Mass &assembler,
			const assembler::AssemblyValsCache &ass_vals_cache,
			const Eigen::MatrixXd &velocity,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver
