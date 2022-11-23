#pragma once

#include <polyfem/Common.hpp>

#include "Assembler.hpp"

#include "Mass.hpp"

#include "Laplacian.hpp"
#include "Bilaplacian.hpp"
#include "Helmholtz.hpp"

#include "LinearElasticity.hpp"
#include "HookeLinearElasticity.hpp"
#include "SaintVenantElasticity.hpp"
#include "NeoHookeanElasticity.hpp"
#include "GenericElastic.hpp"
#include "MooneyRivlinElasticity.hpp"
#include "MultiModel.hpp"
#include "ViscousDamping.hpp"
#include "OgdenElasticity.hpp"

#include "Stokes.hpp"
#include "NavierStokes.hpp"
#include "IncompressibleLinElast.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

#include <vector>
#include <string>

namespace polyfem
{
	namespace assembler
	{
		// factory class that dispaces call to the different assemblers
		// templated with differnt local assemblers
		class AssemblerUtils
		{
		public:
			enum class BasisType
			{
				SIMPLEX_LAGRANGE,
				CUBE_LAGRANGE,
				SPLINE,
				POLY
			};

			typedef std::function<double(const RowVectorNd &, const RowVectorNd &, double, int)> ParamFunc;

			AssemblerUtils();

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
										 utils::SpareMatrixCache &mat_cache,
										 StiffnessMatrix &hessian) const;

			// plotting (eg von mises), assembler is the name of the formulation
			void compute_scalar_value(const std::string &assembler,
									  const int el_id,
									  const basis::ElementBases &bs,
									  const basis::ElementBases &gbs,
									  const Eigen::MatrixXd &local_pts,
									  const Eigen::MatrixXd &fun,
									  Eigen::MatrixXd &result) const;
			// computes tensor, assembler is the name of the formulation
			void compute_tensor_value(const std::string &assembler,
									  const int el_id,
									  const basis::ElementBases &bs,
									  const basis::ElementBases &gbs,
									  const Eigen::MatrixXd &local_pts,
									  const Eigen::MatrixXd &fun,
									  Eigen::MatrixXd &result) const;

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
			void add_multimaterial(const int index, const json &params);
			void set_size(const std::string &assembler, const int dim);
			void init_multimodels(const std::vector<std::string> &materials);

			std::map<std::string, ParamFunc> parameters(const std::string &assembler) const;

			// const LameParameters &lame_params() const { return linear_elasticity_.local_assembler().lame_params(); }
			const Density &density() const { return mass_mat_.local_assembler().density(); }

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
			Assembler<Mass> mass_mat_;
			Assembler<Mass> mass_mat_no_density_;

			Assembler<Laplacian> laplacian_;
			Assembler<Helmholtz> helmholtz_;

			Assembler<BilaplacianMain> bilaplacian_main_;
			MixedAssembler<BilaplacianMixed> bilaplacian_mixed_;
			Assembler<BilaplacianAux> bilaplacian_aux_;

			Assembler<LinearElasticity> linear_elasticity_;
			NLAssembler<LinearElasticity> linear_elasticity_energy_;
			Assembler<HookeLinearElasticity> hooke_linear_elasticity_;

			NLAssembler<SaintVenantElasticity> saint_venant_elasticity_;
			NLAssembler<NeoHookeanElasticity> neo_hookean_elasticity_;
			NLAssembler<GenericElastic<MooneyRivlinElasticity>> mooney_rivlin_elasticity_;
			NLAssembler<MultiModel> multi_models_elasticity_;
			NLAssembler<GenericElastic<OgdenElasticity>> ogden_elasticity_;

			NLAssembler<ViscousDamping> damping_;

			Assembler<StokesVelocity> stokes_velocity_;
			MixedAssembler<StokesMixed> stokes_mixed_;
			Assembler<StokesPressure> stokes_pressure_;

			NLAssembler<NavierStokesVelocity<false>> navier_stokes_velocity_picard_;
			NLAssembler<NavierStokesVelocity<true>> navier_stokes_velocity_;

			Assembler<IncompressibleLinearElasticityDispacement> incompressible_lin_elast_displacement_;
			MixedAssembler<IncompressibleLinearElasticityMixed> incompressible_lin_elast_mixed_;
			Assembler<IncompressibleLinearElasticityPressure> incompressible_lin_elast_pressure_;
		};
	} // namespace assembler
} // namespace polyfem
