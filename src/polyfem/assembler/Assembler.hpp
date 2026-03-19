#pragma once

#include <polyfem/Units.hpp>

#include <polyfem/assembler/AssemblerData.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polyfem/utils/MatrixCache.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Logger.hpp>

#include <MeshFEM/SystemAssembler.hh>

#include <functional>

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
			const double t,
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

	/// abstract class
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
			const double t,
			StiffnessMatrix &stiffness,
			const bool is_mass = false) const { log_and_throw_error("Assembler not implemented by {}!", name()); }

		virtual void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double t,
			NewtonHessian &stiffness,
			const bool is_mass = true) const { log_and_throw_error("Assembler not implemented by {}!", name()); }

		// assemble energy
		virtual double assemble_energy(
			const bool is_volume,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double t,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev) const { log_and_throw_error("Assemble energy not implemented by {}!", name()); }

		virtual Eigen::VectorXd assemble_energy_per_element(
			const bool is_volume,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double t,
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
			const double t,
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
			const double t,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			utils::MatrixCache &mat_cache,
			StiffnessMatrix &grad) const { log_and_throw_error("Assemble hessian not implemented by {}!", name()); }
		
		virtual void assemble_hessian(
			const bool is_volume,
			const int n_basis,
			const bool project_to_psd,
			const double weight,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double t,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			utils::MatrixCache &mat_cache,
			NewtonHessian &H) const { log_and_throw_error("Assemble hessian not implemented by {}!", name()); }

		virtual Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const
			{ log_and_throw_error("Assemble hessian not implemented by {}!", name()); }
		// plotting (eg von mises), assembler is the name of the formulation
		virtual void compute_scalar_value(
			const OutputData &data,
			std::vector<NamedMatrix> &result) const {}

		// computes tensor, assembler is the name of the formulation
		virtual void compute_tensor_value(
			const OutputData &data,
			std::vector<NamedMatrix> &result) const
		{
		}

		// computes tensor, assembler is the name of the formulation
		virtual void compute_stiffness_value(
			const double t,
			const assembler::ElementAssemblyValues &vals,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &tensor) const { log_and_throw_error("Not implemented!"); }

		virtual void compute_dstress_dmu_dlambda(
			const OptAssemblerData &data,
			Eigen::MatrixXd &dstress_dmu,
			Eigen::MatrixXd &dstress_dlambda) const { log_and_throw_adjoint_error("Not implemented!"); }

		virtual void compute_stress_grad_multiply_mat(
			const OptAssemblerData &data,
			const Eigen::MatrixXd &mat,
			Eigen::MatrixXd &stress,
			Eigen::MatrixXd &result) const { log_and_throw_adjoint_error("Not implemented!"); }

		virtual void compute_stress_grad_multiply_stress(
			const OptAssemblerData &data,
			Eigen::MatrixXd &stress,
			Eigen::MatrixXd &result) const
		{
			Eigen::MatrixXd unused;
			compute_stress_grad_multiply_mat(data, Eigen::MatrixXd::Zero(data.grad_u_i.rows(), data.grad_u_i.cols()), stress, unused);
			compute_stress_grad_multiply_mat(data, stress, unused, result);
		}

		virtual void compute_stress_grad_multiply_vect(
			const OptAssemblerData &data,
			const Eigen::MatrixXd &vect,
			Eigen::MatrixXd &stress,
			Eigen::MatrixXd &result) const { log_and_throw_error("Not implemented!"); }

		virtual void compute_stress_grad(
			const OptAssemblerData &data,
			const Eigen::MatrixXd &prev_grad_u_i,
			Eigen::MatrixXd &stress,
			Eigen::MatrixXd &result) const { log_and_throw_adjoint_error("Not implemented!"); }
		virtual void compute_stress_prev_grad(
			const OptAssemblerData &data,
			const Eigen::MatrixXd &prev_grad_u_i,
			Eigen::MatrixXd &result) const { log_and_throw_adjoint_error("Not implemented!"); }

		virtual std::map<std::string, ParamFunc> parameters() const = 0;
		virtual VectorNd compute_rhs(const AutodiffHessianPt &pt) const { log_and_throw_error("Rhs not supported by {}!", name()); }

		virtual Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const { log_and_throw_error("Kernel not supported by {}!", name()); }

		void set_materials(const std::vector<int> &body_ids, const json &body_params, const Units &units, const std::string &root_path);
		virtual void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) {}

		virtual void update_lame_params(const Eigen::MatrixXd &lambdas, const Eigen::MatrixXd &mus)
		{
			log_and_throw_error("Not implemented!");
		}

		virtual bool is_linear() const = 0;
		virtual bool is_solution_displacement() const { return false; }
		virtual bool is_fluid() const { return false; }
		virtual bool is_tensor() const { return false; }

		
		void initSystemAssembler(int n_basis) const {
			m_assembler2D = std::make_unique<SystemAssembler<2>>(n_basis);
			m_assembler3D = std::make_unique<SystemAssembler<3>>(n_basis);
		}

		template<class StencilCallable>
		NewtonHessian buildSparsityPattern(size_t n, StencilCallable &&stencil) const {
			if (size() == 2) { assert(m_assembler2D); return m_assembler2D->sparsityPattern(n, std::forward<StencilCallable>(stencil)); }
			if (size() == 3) { assert(m_assembler3D); return m_assembler3D->sparsityPattern(n, std::forward<StencilCallable>(stencil)); }
			throw std::runtime_error("Unsupported dimension in buildSparsityPattern");
		}

		template<class EvalCallable, class StencilCallable>
		void accumulateHessianContribs(NewtonHessian &H, size_t n, EvalCallable &&eval, StencilCallable &&stencil) const {
			if (size() == 2) { assert(m_assembler2D); m_assembler2D->assembleHessian(H, n, std::forward<EvalCallable>(eval), std::forward<StencilCallable>(stencil)); return; }
			if (size() == 3) { assert(m_assembler3D); m_assembler3D->assembleHessian(H, n, std::forward<EvalCallable>(eval), std::forward<StencilCallable>(stencil)); return; }
			throw std::runtime_error("Unsupported dimension in accumulateHessianContribs");
		}

		mutable std::unique_ptr<SystemAssembler<2>> m_assembler2D;
		mutable std::unique_ptr<SystemAssembler<3>> m_assembler3D;
	protected:
		int size_ = -1;
		
	};

	class MixedNLAssembler : virtual public Assembler
	{
	public:
		using SolutionSplitter = std::function<void(
			const Eigen::MatrixXd &x,
			Eigen::MatrixXd &x_phi,
			Eigen::MatrixXd &x_psi)>;

		using Assembler::assemble_energy;
		using Assembler::assemble_energy_per_element;
		using Assembler::assemble_gradient;
		using Assembler::assemble_hessian;

		virtual ~MixedNLAssembler() = default;

		double assemble_energy(
			const bool is_volume,
			const int n_psi_basis,
			const int n_phi_basis,
			const std::vector<basis::ElementBases> &psi_bases,
			const std::vector<basis::ElementBases> &phi_bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &psi_cache,
			const AssemblyValsCache &phi_cache,
			const double t,
			const double dt,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const SolutionSplitter &split_solution) const;

		Eigen::VectorXd assemble_energy_per_element(
			const bool is_volume,
			const int n_psi_basis,
			const int n_phi_basis,
			const std::vector<basis::ElementBases> &psi_bases,
			const std::vector<basis::ElementBases> &phi_bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &psi_cache,
			const AssemblyValsCache &phi_cache,
			const double t,
			const double dt,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const SolutionSplitter &split_solution) const;

		void assemble_gradient(
			const bool is_volume,
			const int n_psi_basis,
			const int n_phi_basis,
			const std::vector<basis::ElementBases> &psi_bases,
			const std::vector<basis::ElementBases> &phi_bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &psi_cache,
			const AssemblyValsCache &phi_cache,
			const double t,
			const double dt,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const SolutionSplitter &split_solution,
			Eigen::MatrixXd &grad) const;

		void assemble_hessian(
			const bool is_volume,
			const int n_psi_basis,
			const int n_phi_basis,
			const bool project_to_psd,
			const std::vector<basis::ElementBases> &psi_bases,
			const std::vector<basis::ElementBases> &phi_bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &psi_cache,
			const AssemblyValsCache &phi_cache,
			const double t,
			const double dt,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const SolutionSplitter &split_solution,
			utils::MatrixCache &mat_cache,
			StiffnessMatrix &hessian) const;

		bool is_linear() const override { return false; }

	protected:
		virtual int rows() const = 0;
		virtual int cols() const = 0;

		virtual double compute_energy(const MixedNonLinearAssemblerData &data) const = 0;
		virtual Eigen::VectorXd compute_gradient(const MixedNonLinearAssemblerData &data) const = 0;
		virtual Eigen::MatrixXd compute_hessian(const MixedNonLinearAssemblerData &data) const = 0;
	};

	/// Local nonlinear assembler for an arbitrary number of FE spaces. Unlike
	/// NLAssembler and MixedNLAssembler, this class never performs global assembly.
	/// The owning form is responsible for element traversal and gather/scatter.
	class MultiSpacesNLAssembler : virtual public Assembler
	{
	public:
		using Assembler::assemble_gradient;
		using Assembler::assemble_hessian;

		virtual ~MultiSpacesNLAssembler() = default;

		virtual double compute_energy(const MultiSpacesNLAssemblerData &data) const = 0;
		virtual Eigen::VectorXd assemble_gradient(const MultiSpacesNLAssemblerData &data) const = 0;
		virtual Eigen::MatrixXd assemble_hessian(
			const MultiSpacesNLAssemblerData &data,
			int row_space,
			int col_space) const = 0;

		bool is_linear() const override { return false; }
	};

	/// assemble matrix based on the local assembler
	/// local assembler is eg Laplace, LinearElasticity etc
	class LinearAssembler : virtual public Assembler
	{
	public:
		LinearAssembler();
		virtual ~LinearAssembler() = default;

		/// assembles the stiffness matrix for the given basis
		/// the bilinear form (local assembler) is encoded by
		/// the overloaded assemble (see below) function that
		/// the subclass (eg Laplacian) defines
		/// sets stiffness and modifies cache if it has not
		/// already been computed
		void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double t,
			StiffnessMatrix &stiffness,
			const bool is_mass = false) const override;

		void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double t,
			NewtonHessian &stiffness,
			const bool is_mass = true) const override;


		void assembleImpl(
		const bool is_volume,
		const int n_basis,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double t,
		StiffnessMatrix &stiffness,
		const bool is_mass) const;
		
		virtual bool is_linear() const override { return true; }

		/// local assembly function that defines the bilinear form (LHS)
		/// computes and returns a single local stiffness value
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
			const double t,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev) const override;

		// assemble the energy per element
		Eigen::VectorXd assemble_energy_per_element(
			const bool is_volume,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double t,
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
			const double t,
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
			const double t,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			utils::MatrixCache &mat_cache,
			StiffnessMatrix &grad) const override;

		// Integrating MeshFEM's hessian assembly into PolyFEM's NLAssembler interface
		void assemble_hessian(
			const bool is_volume,
			const int n_basis,
			const bool project_to_psd,
			const double weight,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const double t,
			const double dt,
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &displacement_prev,
			utils::MatrixCache &mat_cache,
			NewtonHessian &H) const override;

		

		virtual bool is_linear() const override { return false; }

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

		void set_use_robust_jacobian();

		// plotting (eg von mises), assembler is the name of the formulation
		void compute_scalar_value(
			const OutputData &data,
			std::vector<NamedMatrix> &result) const override
		{
			result.clear();
			Eigen::MatrixXd tmp;
			compute_von_mises_stresses(data, tmp);
			result.emplace_back("von_mises", tmp);
		}

		// computes tensor, assembler is the name of the formulation
		void compute_tensor_value(
			const OutputData &data,
			std::vector<NamedMatrix> &result) const override
		{
			result.clear();
			Eigen::MatrixXd cauchy, pk1, pk2, F;

			compute_stress_tensor(data, ElasticityTensorType::CAUCHY, cauchy);
			compute_stress_tensor(data, ElasticityTensorType::PK1, pk1);
			compute_stress_tensor(data, ElasticityTensorType::PK2, pk2);
			compute_stress_tensor(data, ElasticityTensorType::F, F);

			result.emplace_back("cauchy_stess", cauchy);
			result.emplace_back("pk1_stess", pk1);
			result.emplace_back("pk2_stess", pk2);
			result.emplace_back("F", F);
		}

		void compute_stress_tensor(const OutputData &data,
								   const ElasticityTensorType &type,
								   Eigen::MatrixXd &stresses) const
		{
			assign_stress_tensor(data, size() * size(), type, stresses, [&](const Eigen::MatrixXd &stress) {
				Eigen::MatrixXd tmp = stress;
				auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
				return Eigen::MatrixXd(a);
			});
		}

		void compute_von_mises_stresses(const OutputData &data,
										Eigen::MatrixXd &stresses) const
		{
			assign_stress_tensor(data, 1, ElasticityTensorType::CAUCHY, stresses, [&](const Eigen::MatrixXd &stress) {
				Eigen::Matrix<double, 1, 1> res;
				res.setConstant(von_mises_stress_for_stress_tensor(stress));
				return res;
			});
		}

		bool is_solution_displacement() const override { return true; }
		bool is_tensor() const override { return true; }
		virtual bool allow_inversion() const = 0;

		virtual void assign_stress_tensor(const OutputData &data,
										  const int all_size,
										  const ElasticityTensorType &type,
										  Eigen::MatrixXd &all,
										  const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const = 0;

	protected:
		bool use_robust_jacobian = false;
	};

	class ElasticityNLAssembler : virtual public ElasticityAssembler, virtual public NLAssembler
	{
	};

	struct ElementBasisStencil {
    	ElementBasisStencil(const std::vector<basis::ElementBases> &bases) : m_b(bases) { }
    	std::vector<int> operator()(int e) const { 
			std::vector<int> nodesIndices; 
			const int n_loc_bases = int(m_b[e].bases.size());		
			for (int i = 0; i < n_loc_bases; ++i)
			{
				const auto global_i = m_b[e].bases[i].global();
				for (size_t ii = 0; ii < global_i.size(); ++ii)
				{
					nodesIndices.push_back(global_i[ii].index);
				} // size of global_i is mostly 1
			}
			return nodesIndices;
		 }
	private:
    	const std::vector<basis::ElementBases> &m_b;
	};

	struct ElementHessianEvaluator{

		ElementHessianEvaluator(const Assembler &assembler, const bool is_volume, const bool project_to_psd, const double weight, const Eigen::VectorXd &x, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const AssemblyValsCache &cache, const double t, const double dt, const Eigen::MatrixXd &displacement_prev)
			: m_assembler(assembler), m_is_volume(is_volume), m_project_to_psd(project_to_psd), m_weight(weight), m_x(x), m_bases(bases), m_gbases(gbases), m_cache(cache), m_t(t), m_dt(dt), m_displacement_prev(displacement_prev) {}

		Eigen::MatrixXd operator()(size_t e) const {
			ElementAssemblyValues vals;
			m_cache.compute(e, m_is_volume, m_bases[e], m_gbases[e], vals);

			const quadrature::Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			QuadratureVector qVec = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());

			auto stiffness_val = m_assembler.assemble_hessian(NonLinearAssemblerData(vals, m_t, m_dt, m_x, m_displacement_prev, qVec));
			assert(stiffness_val.rows() == n_loc_bases * m_assembler.size());
			assert(stiffness_val.cols() == n_loc_bases * m_assembler.size());

			if (m_project_to_psd)
				stiffness_val = ipc::project_to_psd(stiffness_val);

			return stiffness_val * m_weight;
			
		}
		private:
			const Assembler &m_assembler;
			const bool m_is_volume;
			const bool m_project_to_psd;
			const double m_weight;
			const Eigen::VectorXd &m_x;
			const std::vector<basis::ElementBases> &m_bases;
			const std::vector<basis::ElementBases> &m_gbases;
			const AssemblyValsCache &m_cache;
			const double m_t;
			const double m_dt;
			const Eigen::MatrixXd &m_displacement_prev;

	};



} // namespace polyfem::assembler
