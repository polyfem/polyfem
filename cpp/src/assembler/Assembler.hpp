#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include "ElementAssemblyValues.hpp"

#include "Problem.hpp"

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

namespace poly_fem
{
	template<class LocalAssembler>
	class Assembler
	{
	public:
		void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			Eigen::SparseMatrix<double> &stiffness) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};


	template<class LocalAssembler>
	class NLAssembler
	{
	public:
		void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &rhs) const;

		void assemble_grad(
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement,
			Eigen::SparseMatrix<double> &grad) const;

		double compute_energy(
			const bool is_volume,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};
}

#endif //ASSEMBLER_HPP
