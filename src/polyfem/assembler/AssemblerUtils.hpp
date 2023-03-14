#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::assembler
{

	class AssemblerUtils
	{
	public:
		static std::string other_assembler_name(const std::string &formulation);

		static std::shared_ptr<Assembler> make_assembler(const std::string &formulation);
		static std::shared_ptr<MixedAssembler> make_mixed_assembler(const std::string &formulation);

		enum class BasisType
		{
			SIMPLEX_LAGRANGE,
			CUBE_LAGRANGE,
			SPLINE,
			POLY
		};

		AssemblerUtils() = delete;

		// utility to merge 3 blocks of mixed matrices, A=velocity_stiffness, B=mixed_stiffness, and C=pressure_stiffness
		//  A   B
		//  B^T C
		static void merge_mixed_matrices(
			const int n_bases, const int n_pressure_bases, const int problem_dim, const bool add_average,
			const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
			StiffnessMatrix &stiffness);

		static int quadrature_order(const std::string &assembler, const int basis_degree, const BasisType &b_type, const int dim);
	};
} // namespace polyfem::assembler
