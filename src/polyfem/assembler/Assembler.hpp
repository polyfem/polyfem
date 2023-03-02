#pragma once

#include <polyfem/assembler/AssemblerData.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

// this casses are instantiated in the cpp, cannot be used with generic assembler
// without adding template instantiation
namespace polyfem::assembler
{
	class Assembler
	{
	public:
		virtual ~Assembler() = default;

		int size() const { return size_; }
		virtual void set_size(const int size) { size_ = size; }

	protected:
		int size_ = -1;
	};

	// assemble matrix based on the local assembler
	// local assembler is eg Laplce, LinearElasticy etc
	template <typename LocalBlockMatrix>
	class LinearAssembler : virtual public Assembler
	{
	public:
		LinearAssembler();
		virtual ~LinearAssembler() = default;

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

	using ScalarLinearAssembler = LinearAssembler<Eigen::Matrix<double, 1, 1>>;
	using TensorLinearAssembler = LinearAssembler<Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>>;

	// mixed formulation assembler
	template <typename LocalBlockMatrix>
	class MixedAssembler : virtual public Assembler
	{
	public:
		MixedAssembler();
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

	using ScalarMixedAssembler = MixedAssembler<Eigen::Matrix<double, 1, 1>>;
	using TensorMixedAssembler = MixedAssembler<Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>>;

	// non-linear assembler (eg neohookean elasticity)
	class NLAssembler : virtual public Assembler
	{
	public:
		virtual ~NLAssembler() = default;

		// assemble energy
		double assemble_energy(
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
