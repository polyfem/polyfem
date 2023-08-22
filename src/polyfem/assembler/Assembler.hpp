#pragma once

#include <polyfem/Units.hpp>

#include <polyfem/assembler/AssemblerData.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Logger.hpp>

// this casses are instantiated in the cpp, cannot be used with generic assembler
// without adding template instantiation
namespace polyfem::assembler
{
	// mixed formulation assembler
	class MixedAssembler
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

		virtual std::string name() const = 0;

		int size() const { return size_; }
		virtual void set_size(const int size) { size_ = size; }

	protected:
		int size_ = -1;

		virtual int rows() const = 0;
		virtual int cols() const = 0;

		virtual Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> assemble(const MixedAssemblerData &data) const = 0;
	};

	class Assembler
	{
	public:
		typedef std::pair<std::string, Eigen::MatrixXd> NamedMatrix;
		typedef std::function<double(const RowVectorNd &, const RowVectorNd &, double, int)> ParamFunc;

		virtual ~Assembler() = default;

		virtual std::string name() const = 0;

		int size() const { return size_; }
		virtual void set_size(const int size) { size_ = size; }

		// assembler stiffness matrix, is the mesh is volumetric, number of bases and bases (FE and geom)
		// gbases and bases can be the same (ie isoparametric)
		virtual void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			StiffnessMatrix &stiffness,
			const bool is_mass = false) const { log_and_throw_error("Assembler not implemented by {}!", name()); }

		// assemble energy
		virtual double assemble_energy(
			const bool is_volume,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev) const { log_and_throw_error("Assemble energy not implemented by {}!", name()); }

		// assemble gradient of energy (rhs)
		virtual void assemble_gradient(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			Eigen::MatrixXd &rhs) const { log_and_throw_error("Assemble grad not implemented by {}!", name()); }

		// assemble hessian of energy (grad)
		virtual void assemble_hessian(
			const bool is_volume,
			const int n_basis,
			const bool project_to_psd,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			utils::SparseMatrixCache &mat_cache,
			StiffnessMatrix &grad) const { log_and_throw_error("Assemble hessian not implemented by {}!", name()); }

		// plotting (eg von mises), assembler is the name of the formulation
		virtual void compute_scalar_value(
			const int el_id,
			const basis::ElementBases &bs,
			const basis::ElementBases &gbs,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun,
			std::vector<NamedMatrix> &result) const {}

		// computes tensor, assembler is the name of the formulation
		virtual void compute_tensor_value(
			const int el_id,
			const basis::ElementBases &bs,
			const basis::ElementBases &gbs,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun,
			std::vector<NamedMatrix> &result) const {}

		// computes tensor, assembler is the name of the formulation
		virtual void compute_stiffness_value(
			const assembler::ElementAssemblyValues &vals,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &tensor) const { log_and_throw_error("Not implemented!"); }

		virtual void compute_dstress_dmu_dlambda(
			const int el_id,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &global_pts,
			const Eigen::MatrixXd &grad_u_i,
			Eigen::MatrixXd &dstress_dmu,
			Eigen::MatrixXd &dstress_dlambda) const { log_and_throw_error("Not implemented!"); }

		virtual void compute_stress_grad_multiply_mat(
			const int el_id,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &global_pts,
			const Eigen::MatrixXd &grad_u_i,
			const Eigen::MatrixXd &mat,
			Eigen::MatrixXd &stress,
			Eigen::MatrixXd &result) const { log_and_throw_error("Not implemented!"); }

		virtual void compute_stress_grad_multiply_stress(
			const int el_id,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &global_pts,
			const Eigen::MatrixXd &grad_u_i,
			Eigen::MatrixXd &stress,
			Eigen::MatrixXd &result) const
		{
			Eigen::MatrixXd unused;
			compute_stress_grad_multiply_mat(el_id, local_pts, global_pts, grad_u_i, Eigen::MatrixXd::Zero(grad_u_i.rows(), grad_u_i.cols()), stress, unused);
			compute_stress_grad_multiply_mat(el_id, local_pts, global_pts, grad_u_i, stress, unused, result);
		}

		virtual void compute_stress_grad_multiply_vect(
			const int el_id,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &global_pts,
			const Eigen::MatrixXd &grad_u_i,
			const Eigen::MatrixXd &vect,
			Eigen::MatrixXd &stress,
			Eigen::MatrixXd &result) const { log_and_throw_error("Not implemented!"); }

		virtual void compute_stress_grad(
			const int el_id,
			const double dt,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &global_pts,
			const Eigen::MatrixXd &grad_u_i,
			const Eigen::MatrixXd &prev_grad_u_i,
			Eigen::MatrixXd &stress,
			Eigen::MatrixXd &result) const { log_and_throw_error("Not implemented!"); }
		virtual void compute_stress_prev_grad(
			const int el_id,
			const double dt,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &global_pts,
			const Eigen::MatrixXd &grad_u_i,
			const Eigen::MatrixXd &prev_grad_u_i,
			Eigen::MatrixXd &result) const { log_and_throw_error("Not implemented!"); }

		virtual std::map<std::string, ParamFunc> parameters() const = 0;
		virtual VectorNd compute_rhs(const AutodiffHessianPt &pt) const { log_and_throw_error("Rhs not supported by {}!", name()); }

		virtual Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const { log_and_throw_error("Kernel not supported by {}!", name()); }

		void set_materials(const std::vector<int> &body_ids, const json &body_params, const Units &units);
		virtual void add_multimaterial(const int index, const json &params, const Units &units) {}

		virtual void update_lame_params(const Eigen::MatrixXd &lambdas, const Eigen::MatrixXd &mus)
		{
			log_and_throw_error("Not implemented!");
		}

		virtual bool is_linear() const = 0;
		virtual bool is_solution_displacement() const { return false; }
		virtual bool is_fluid() const { return false; }
		virtual bool is_tensor() const { return false; }

	protected:
		int size_ = -1;
	};

	// assemble matrix based on the local assembler
	// local assembler is eg Laplce, LinearElasticy etc
	class LinearAssembler : virtual public Assembler
	{
	public:
		LinearAssembler();
		virtual ~LinearAssembler() = default;

		void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			StiffnessMatrix &stiffness,
			const bool is_mass = false) const override;

		virtual bool is_linear() const override { return true; }

		virtual Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> assemble(const LinearAssemblerData &data) const = 0;
	};

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
			const Eigen::MatrixXd &displacement_prev) const override;

		// assemble gradient of energy (rhs)
		void assemble_gradient(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			Eigen::MatrixXd &rhs) const override;

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
			utils::SparseMatrixCache &mat_cache,
			StiffnessMatrix &grad) const override;

		virtual bool is_linear() const override { return false; }

	protected:
		// energy, gradient, and hessian used in newton method
		virtual double compute_energy(const NonLinearAssemblerData &data) const = 0;
		virtual Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const = 0;
		virtual Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const = 0;
	};

	class ElasticityAssembler : virtual public Assembler
	{
	public:
		ElasticityAssembler() {}
		virtual ~ElasticityAssembler() = default;

		// plotting (eg von mises), assembler is the name of the formulation
		void compute_scalar_value(
			const int el_id,
			const basis::ElementBases &bs,
			const basis::ElementBases &gbs,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun,
			std::vector<NamedMatrix> &result) const override
		{
			result.clear();
			Eigen::MatrixXd tmp;
			compute_von_mises_stresses(el_id, bs, gbs, local_pts, fun, tmp);
			result.emplace_back("von_mises", tmp);
		}

		// computes tensor, assembler is the name of the formulation
		void compute_tensor_value(
			const int el_id,
			const basis::ElementBases &bs,
			const basis::ElementBases &gbs,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun,
			std::vector<NamedMatrix> &result) const override
		{
			result.clear();
			Eigen::MatrixXd cauchy, pk1, pk2, F;

			compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::CAUCHY, cauchy);
			compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK1, pk1);
			compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::PK2, pk2);
			compute_stress_tensor(el_id, bs, gbs, local_pts, fun, ElasticityTensorType::F, F);

			result.emplace_back("cauchy_stess", cauchy);
			result.emplace_back("pk1_stess", pk1);
			result.emplace_back("pk2_stess", pk2);
			result.emplace_back("F", F);
		}

		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const ElasticityTensorType &type, Eigen::MatrixXd &stresses) const
		{
			assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size() * size(), type, stresses, [&](const Eigen::MatrixXd &stress) {
				Eigen::MatrixXd tmp = stress;
				auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
				return Eigen::MatrixXd(a);
			});
		}

		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
		{
			assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, ElasticityTensorType::CAUCHY, stresses, [&](const Eigen::MatrixXd &stress) {
				Eigen::Matrix<double, 1, 1> res;
				res.setConstant(von_mises_stress_for_stress_tensor(stress));
				return res;
			});
		}

		bool is_solution_displacement() const override { return true; }
		bool is_tensor() const override { return true; }

	protected:
		virtual void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const = 0;
	};
} // namespace polyfem::assembler
