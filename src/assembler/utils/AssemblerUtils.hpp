#pragma once



#include <polyfem/Common.hpp>

#include <polyfem/Assembler.hpp>
#include <polyfem/MassMatrixAssembler.hpp>

#include <polyfem/Laplacian.hpp>
#include <polyfem/Bilaplacian.hpp>
#include <polyfem/Helmholtz.hpp>

#include <polyfem/LinearElasticity.hpp>
#include <polyfem/HookeLinearElasticity.hpp>
#include <polyfem/SaintVenantElasticity.hpp>
#include <polyfem/NeoHookeanElasticity.hpp>
// #include <polyfem/OgdenElasticity.hpp>

#include <polyfem/Stokes.hpp>
#include <polyfem/IncompressibleLinElast.hpp>

#include <polyfem/ProblemWithSolution.hpp>

#include <vector>

namespace polyfem
{
	class AssemblerUtils
	{
	private:
		AssemblerUtils();

	public:
		static AssemblerUtils &instance();

		//Linear
		void assemble_problem(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			Eigen::SparseMatrix<double> &stiffness) const;

		void assemble_mass_matrix(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			Eigen::SparseMatrix<double> &mass) const;

		void assemble_mixed_problem(const std::string &assembler,
			const bool is_volume,
			const int n_psi_basis,
			const int n_phi_basis,
			const std::vector< ElementBases > &psi_bases,
			const std::vector< ElementBases > &phi_bases,
			const std::vector< ElementBases > &gbases,
			Eigen::SparseMatrix<double> &stiffness) const;

		void assemble_pressure_problem(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			Eigen::SparseMatrix<double> &stiffness) const;


		//Non linear
		double assemble_energy(const std::string &assembler,
			const bool is_volume,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement) const;

		void assemble_energy_gradient(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &grad) const;

		void assemble_energy_hessian(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement,
			Eigen::SparseMatrix<double> &hessian) const;


		//plotting
		void compute_scalar_value(const std::string &assembler,
			const ElementBases &bs,
			const ElementBases &gbs,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun,
			Eigen::MatrixXd &result) const;

		void compute_tensor_value(const std::string &assembler,
			const ElementBases &bs,
			const ElementBases &gbs,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun,
			Eigen::MatrixXd &result) const;

		//for errors
		VectorNd compute_rhs(const std::string &assembler, const AutodiffHessianPt &pt) const;

		//for constraints
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		local_assemble(const std::string &assembler,
			const ElementAssemblyValues &vals,
			const int i,
			const int j,
			const QuadratureVector &da) const;

		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const std::string &assembler, const int dim, const AutodiffScalarGrad &r) const;

		//aux
		void set_parameters(const json &params);

		bool is_linear(const std::string &assembler) const;

		bool is_solution_displacement(const std::string &assembler) const;

		bool is_scalar(const std::string &assembler) const;
		bool is_tensor(const std::string &assembler) const;
		bool is_mixed(const std::string &assembler) const;

		//getters
		const std::vector<std::string> &scalar_assemblers() const { return scalar_assemblers_; }
		const std::vector<std::string> &tensor_assemblers() const { return tensor_assemblers_; }
		// const std::vector<std::string> &mixed_assemblers() const { return mixed_assemblers_; }

		void clear_cache();

	private:
		MassMatrixAssembler mass_mat_assembler_;
		Assembler<Laplacian> laplacian_;
		Assembler<Helmholtz> helmholtz_;

		Assembler<BilaplacianMain> bilaplacian_main_;
		MixedAssembler<BilaplacianMixed> bilaplacian_mixed_;
		Assembler<BilaplacianAux> bilaplacian_aux_;

		Assembler<LinearElasticity> linear_elasticity_;
		Assembler<HookeLinearElasticity> hooke_linear_elasticity_;

		NLAssembler<SaintVenantElasticity> saint_venant_elasticity_;
		NLAssembler<NeoHookeanElasticity> neo_hookean_elasticity_;
		// NLAssembler<OgdenElasticity> ogden_elasticity_;

		Assembler<StokesVelocity> stokes_velocity_;
		MixedAssembler<StokesMixed> stokes_mixed_;
		Assembler<StokesPressure> stokes_pressure_;

		Assembler<IncompressibleLinearElasticityDispacement> incompressible_lin_elast_displacement_;
		MixedAssembler<IncompressibleLinearElasticityMixed> incompressible_lin_elast_mixed_;
		Assembler<IncompressibleLinearElasticityPressure> incompressible_lin_elast_pressure_;

		std::vector<std::string> scalar_assemblers_;
		std::vector<std::string> tensor_assemblers_;
		std::vector<std::string> mixed_assemblers_;
	};
}
