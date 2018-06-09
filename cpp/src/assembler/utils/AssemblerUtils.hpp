#pragma once

#include "Common.hpp"

#include "Assembler.hpp"

#include "Laplacian.hpp"
#include "Helmholtz.hpp"

#include "LinearElasticity.hpp"
#include "HookeLinearElasticity.hpp"
#include "SaintVenantElasticity.hpp"
#include "NeoHookeanElasticity.hpp"
#include "OgdenElasticity.hpp"

#include "ProblemWithSolution.hpp"

#include <vector>

namespace poly_fem
{
	class AssemblerUtils
	{
	private:
		AssemblerUtils();

	public:
		static AssemblerUtils &instance();

		//Linear
		void assemble_scalar_problem(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			Eigen::SparseMatrix<double> &stiffness);

		void assemble_tensor_problem(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			Eigen::SparseMatrix<double> &stiffness);


		//Non linear
		double assemble_tensor_energy(const std::string &assembler,
			const bool is_volume,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement);

		void assemble_tensor_energy_gradient(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &grad);

		void assemble_tensor_energy_hessian(const std::string &assembler,
			const bool is_volume,
			const int n_basis,
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const Eigen::MatrixXd &displacement,
			Eigen::SparseMatrix<double> &hessian);


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

		Eigen::Matrix<AutodiffPt, Eigen::Dynamic, 1, 0, 3, 1> kernel(const std::string &assembler, const int dim, const AutodiffPt &r) const;

		//aux
		void set_parameters(const json &params);

		bool is_linear(const std::string &assembler) const;

		//getters
		const std::vector<std::string> &scalar_assemblers() const { return scalar_assemblers_; }
		const std::vector<std::string> &tensor_assemblers() const { return tensor_assemblers_; }

		void clear_cache();

	private:
		Assembler<Laplacian> laplacian_;
		Assembler<Helmholtz> helmholtz_;

		Assembler<LinearElasticity> linear_elasticity_;
		Assembler<HookeLinearElasticity> hooke_linear_elasticity_;

		NLAssembler<SaintVenantElasticity> saint_venant_elasticity_;
		NLAssembler<NeoHookeanElasticity> neo_hookean_elasticity_;
		NLAssembler<OgdenElasticity> ogden_elasticity_;

		std::vector<std::string> scalar_assemblers_;
		std::vector<std::string> tensor_assemblers_;
	};
}
