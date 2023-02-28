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
	class GenericAssembler
	{
	public:
		virtual ~GenericAssembler() = default;

		int size() const { return size_; }
		void set_size(const int size) { size_ = size; }

	protected:
		int size_ = -1;
	};

	// assemble matrix based on the local assembler
	// local assembler is eg Laplce, LinearElasticy etc
	template <typename LocalBlockMatrix>
	class Assembler : public GenericAssembler // TODO: rename to LinearAssembler
	{
	public:
		Assembler();
		virtual ~Assembler() = default;

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

	protected:
		virtual LocalBlockMatrix assemble(const LinearAssemblerData &data) const = 0;
	};

	using ScalarAssembler = Assembler<Eigen::Matrix<double, 1, 1>>;
	using TensorAssembler = Assembler<Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>>;

	// mixed formulation assembler
	template <typename LocalBlockMatrix>
	class MixedAssembler : public GenericAssembler
	{
	public:
		virtual ~MixedAssembler() = default;

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

	protected:
		virtual int rows() const = 0;
		virtual int cols() const = 0;

		virtual LocalBlockMatrix assemble(const MixedAssemblerData &data) const = 0;
	};

	// non-linear assembler (eg neohookean elasticity)
	class NLAssembler : public GenericAssembler
	{
	public:
		virtual ~NLAssembler() = default;

		// assemble energy
		double assemble(
			const bool is_volume,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev) const;

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

	protected:
		// energy, gradient, and hessian used in newton method
		virtual double compute_energy(const NonLinearAssemblerData &data) const = 0;
		virtual Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const = 0;
		virtual Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const = 0;
	};
} // namespace polyfem::assembler
