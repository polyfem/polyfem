#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include <polyfem/ElementAssemblyValues.hpp>

#include <polyfem/Problem.hpp>

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

namespace polyfem
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
			StiffnessMatrix &stiffness) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};


	template<class LocalAssembler>
	class MixedAssembler
	{
	public:
		void assemble(
			const bool is_volume,
			const int n_psi_basis,
			const int n_phi_basis,
			const std::vector< ElementBases > &psi_bases,
			const std::vector< ElementBases > &phi_bases,
			const std::vector< ElementBases > &gbases,
			StiffnessMatrix &stiffness) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};


	template<class LocalAssembler>
	class NLAssembler
	{
	public:
		void assemble_grad(
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &rhs) const;

		void assemble_hessian(
			const bool is_volume,
			const int n_basis,
			const bool project_to_psd,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const Eigen::MatrixXd &displacement,
			StiffnessMatrix &grad) const;

		double assemble(
			const bool is_volume,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

		void clear_cache() { }

	private:
		LocalAssembler local_assembler_;
	};
}

#endif //ASSEMBLER_HPP
