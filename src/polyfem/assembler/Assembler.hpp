#pragma once

#include "AssemblyValsCache.hpp"

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

// this casses are instantiated in the cpp, cannot be used with generic assembler
// without adding template instantiation
namespace polyfem::assembler
{
	// assemble matrix based on the local assembler
	// local assembler is eg Laplce, LinearElasticy etc
	template <class LocalAssembler>
	class Assembler
	{
	public:
		// assembler stiffness matrix, is the mesh is volumetric, number of bases and bases (FE and geom)
		// gbases and bases can be the same (ie isoparametric)
		void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			StiffnessMatrix &stiffness,
			const bool is_mass = false) const;

		// references to local assemblers
		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};

	// mixed formulation assembler
	template <class LocalAssembler>
	class MixedAssembler
	{
	public:
		// this assembler takes two bases: psi_bases are the scalar ones, phi_bases are the tensor ones
		// both have the same geometric mapping
		void assemble(
			const bool is_volume,
			const int n_psi_basis,
			const int n_phi_basis,
			const std::vector<basis::ElementBases> &psi_bases,
			const std::vector<basis::ElementBases> &phi_bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &psi_cache,
			const AssemblyValsCache &phi_cache,
			StiffnessMatrix &stiffness) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};

	// non-linear assembler (eg neohookean elasticity)
	template <class LocalAssembler>
	class NLAssembler
	{
	public:
		// assemble gradient of energy (rhs)
		void assemble_grad(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			Eigen::MatrixXd &rhs) const;

		// assemble hessian of energy (grad)
		void assemble_hessian(
			const bool is_volume,
			const int n_basis,
			const bool project_to_psd,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			utils::SpareMatrixCache &mat_cache,
			StiffnessMatrix &grad) const;

		// assemble energy
		double assemble(
			const bool is_volume,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};
} // namespace polyfem::assembler
