#pragma once

#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::assembler
{
	class Mass;
	class Density;
	class Laplacian;
	class Helmholtz;
	class BilaplacianMain;
	class BilaplacianMixed;
	class BilaplacianAux;
	class LinearElasticity;
	class HookeLinearElasticity;
	class SaintVenantElasticity;
	class NeoHookeanElasticity;
	class MooneyRivlinElasticity;
	class MultiModel;
	class UnconstrainedOgdenElasticity;
	class IncompressibleOgdenElasticity;
	class ViscousDamping;
	class StokesVelocity;
	class StokesMixed;
	class StokesPressure;
	template <bool full_gradient>
	class NavierStokesVelocity;
	class IncompressibleLinearElasticityDispacement;
	class IncompressibleLinearElasticityMixed;
	class IncompressibleLinearElasticityPressure;

	// factory class that dispaces call to the different assemblers
	// templated with differnt local assemblers
	class AssemblerUtils
	{
	public:
		typedef std::pair<std::string, Eigen::MatrixXd> NamedMatrix;

		enum class BasisType
		{
			SIMPLEX_LAGRANGE,
			CUBE_LAGRANGE,
			SPLINE,
			POLY
		};

		typedef std::function<double(const RowVectorNd &, const RowVectorNd &, double, int)> ParamFunc;

		AssemblerUtils();
		~AssemblerUtils();

		// Linear, assembler is the name of the formulation
		void assemble_problem(const std::string &assembler,
							  const bool is_volume,
							  const int n_basis,
							  const std::vector<basis::ElementBases> &bases,
							  const std::vector<basis::ElementBases> &gbases,
							  const AssemblyValsCache &cache,
							  StiffnessMatrix &stiffness) const;

		// mass matrix assembler, assembler is the name of the formulation
		void assemble_mass_matrix(const std::string &assembler,
								  const bool is_volume,
								  const int n_basis,
								  const bool use_density,
								  const std::vector<basis::ElementBases> &bases,
								  const std::vector<basis::ElementBases> &gbases,
								  const AssemblyValsCache &cache,
								  StiffnessMatrix &mass) const;

		// mixed assembler phi is the tensor, psi the scalar, assembler is the name of the formulation
		void assemble_mixed_problem(const std::string &assembler,
									const bool is_volume,
									const int n_psi_basis,
									const int n_phi_basis,
									const std::vector<basis::ElementBases> &psi_bases,
									const std::vector<basis::ElementBases> &phi_bases,
									const std::vector<basis::ElementBases> &gbases,
									const AssemblyValsCache &psi_cache,
									const AssemblyValsCache &phi_cache,
									StiffnessMatrix &stiffness) const;
		// pressure pressure assembler, assembler is the name of the formulation
		void assemble_pressure_problem(const std::string &assembler,
									   const bool is_volume,
									   const int n_basis,
									   const std::vector<basis::ElementBases> &bases,
									   const std::vector<basis::ElementBases> &gbases,
									   const AssemblyValsCache &cache,
									   StiffnessMatrix &stiffness) const;

		// Non linear energy, assembler is the name of the formulation
		double assemble_energy(const std::string &assembler,
							   const bool is_volume,
							   const std::vector<basis::ElementBases> &bases,
							   const std::vector<basis::ElementBases> &gbases,
							   const AssemblyValsCache &cache,
							   const double dt,
							   const Eigen::MatrixXd &displacement,
							   const Eigen::MatrixXd &displacement_prev) const;

		// non linear gradient, assembler is the name of the formulation
		void assemble_energy_gradient(const std::string &assembler,
									  const bool is_volume,
									  const int n_basis,
									  const std::vector<basis::ElementBases> &bases,
									  const std::vector<basis::ElementBases> &gbases,
									  const AssemblyValsCache &cache,
									  const double dt,
									  const Eigen::MatrixXd &displacement,
									  const Eigen::MatrixXd &displacement_prev,
									  Eigen::MatrixXd &grad) const;
		// non-linear hessian, assembler is the name of the formulation
		void assemble_energy_hessian(const std::string &assembler,
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
									 StiffnessMatrix &hessian) const;

		// plotting (eg von mises), assembler is the name of the formulation
		void compute_scalar_value(const std::string &assembler,
								  const int el_id,
								  const basis::ElementBases &bs,
								  const basis::ElementBases &gbs,
								  const Eigen::MatrixXd &local_pts,
								  const Eigen::MatrixXd &fun,
								  std::vector<NamedMatrix> &result) const;
		// computes tensor, assembler is the name of the formulation
		void compute_tensor_value(const std::string &assembler,
								  const int el_id,
								  const basis::ElementBases &bs,
								  const basis::ElementBases &gbs,
								  const Eigen::MatrixXd &local_pts,
								  const Eigen::MatrixXd &fun,
								  std::vector<NamedMatrix> &result) const;

		// for errors, uses the rhs methods inside local assemblers
		VectorNd compute_rhs(const std::string &assembler, const AutodiffHessianPt &pt) const;

		// for constraints in polygonal bases
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		local_assemble(const std::string &assembler,
					   const ElementAssemblyValues &vals,
					   const int i,
					   const int j,
					   const QuadratureVector &da) const;

		// returns the kernel of the assembler, if present
		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const std::string &assembler, const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const;

		// dispaces to all set parameters of the local assemblers
		void set_materials(const std::vector<int> &body_ids, const json &body_params);
		void add_multimaterial(const int index, const json &params);
		void set_size(const std::string &assembler, const int dim);
		void init_multimodels(const std::vector<std::string> &materials);

		std::map<std::string, ParamFunc> parameters(const std::string &assembler) const;

		// const LameParameters &lame_params() const { return linear_elasticity_.local_assembler().lame_params(); }
		const Density &density() const;

		// checks if assembler is linear
		static bool is_linear(const std::string &assembler);

		// checks if assembler solution is displacement (true for elasticty)
		static bool is_solution_displacement(const std::string &assembler);

		// checks if assembler is scalar (Laplace and Helmolz)
		static bool is_scalar(const std::string &assembler);
		// checks if assembler is tensor (other)
		static bool is_tensor(const std::string &assembler);
		// checks if assembler is mixed (eg, stokes)
		static bool is_mixed(const std::string &assembler);
		// checks if it is a fluid simulation
		static bool is_fluid(const std::string &assembler);

		bool has_damping() const;

		// gets the names of all assemblers
		static std::vector<std::string> scalar_assemblers();
		static std::vector<std::string> tensor_assemblers();
		// const std::vector<std::string> &mixed_assemblers() const { return mixed_assemblers_; }

		// utility to merge 3 blocks of mixed matrices, A=velocity_stiffness, B=mixed_stiffness, and C=pressure_stiffness
		//  A   B
		//  B^T C
		static void merge_mixed_matrices(
			const int n_bases, const int n_pressure_bases, const int problem_dim, const bool add_average,
			const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
			StiffnessMatrix &stiffness);

		static int quadrature_order(const std::string &assembler, const int basis_degree, const BasisType &b_type, const int dim);

	private:
		// all assemblers
		std::unique_ptr<Mass> mass_mat_;
		std::unique_ptr<Mass> mass_mat_no_density_;

		std::unique_ptr<Laplacian> laplacian_;
		std::unique_ptr<Helmholtz> helmholtz_;

		std::unique_ptr<BilaplacianMain> bilaplacian_main_;
		std::unique_ptr<BilaplacianMixed> bilaplacian_mixed_;
		std::unique_ptr<BilaplacianAux> bilaplacian_aux_;

		std::unique_ptr<LinearElasticity> linear_elasticity_;
		std::unique_ptr<HookeLinearElasticity> hooke_linear_elasticity_;

		std::unique_ptr<SaintVenantElasticity> saint_venant_elasticity_;
		std::unique_ptr<NeoHookeanElasticity> neo_hookean_elasticity_;
		std::unique_ptr<MooneyRivlinElasticity> mooney_rivlin_elasticity_;
		std::unique_ptr<MultiModel> multi_models_elasticity_;
		std::unique_ptr<UnconstrainedOgdenElasticity> unconstrained_ogden_elasticity_;
		std::unique_ptr<IncompressibleOgdenElasticity> incompressible_ogden_elasticity_;

		std::unique_ptr<ViscousDamping> damping_;

		std::unique_ptr<StokesVelocity> stokes_velocity_;
		std::unique_ptr<StokesMixed> stokes_mixed_;
		std::unique_ptr<StokesPressure> stokes_pressure_;

		std::unique_ptr<NavierStokesVelocity<false>> navier_stokes_velocity_picard_;
		std::unique_ptr<NavierStokesVelocity<true>> navier_stokes_velocity_;

		std::unique_ptr<IncompressibleLinearElasticityDispacement> incompressible_lin_elast_displacement_;
		std::unique_ptr<IncompressibleLinearElasticityMixed> incompressible_lin_elast_mixed_;
		std::unique_ptr<IncompressibleLinearElasticityPressure> incompressible_lin_elast_pressure_;
	};
} // namespace polyfem::assembler
