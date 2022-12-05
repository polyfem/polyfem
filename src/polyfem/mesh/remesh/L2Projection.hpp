#pragma once

#include <polyfem/assembler/MassMatrixAssembler.hpp>
#include <polyfem/solver/NLProblem.hpp>

namespace polyfem::mesh
{
	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const bool is_volume,
		const int size,
		// Project from these bases
		const int n_from_basis,
		const std::vector<polyfem::basis::ElementBases> &from_bases,
		const std::vector<polyfem::basis::ElementBases> &from_gbases,
		// to these bases
		const int n_to_basis,
		const std::vector<polyfem::basis::ElementBases> &to_bases,
		const std::vector<polyfem::basis::ElementBases> &to_gbases,
		// with these boundary values.
		const std::vector<int> &boundary_nodes,
		const Obstacle &obstacle,
		const Eigen::MatrixXd &target_x,
		// DOF
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x,
		const bool lump_mass_matrix = false);

	class L2ProjectionForm : public polyfem::solver::Form
	{
	public:
		L2ProjectionForm(
			const StiffnessMatrix &M,
			const StiffnessMatrix &A,
			const Eigen::VectorXd &x_prev);

	protected:
		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

		StiffnessMatrix M_;
		Eigen::VectorXd rhs_;
	};

	class StaticBoundaryNLProblem : public polyfem::solver::NLProblem
	{
	public:
		StaticBoundaryNLProblem(
			const int full_size,
			const std::vector<int> &boundary_nodes,
			const Eigen::VectorXd &boundary_values,
			const std::vector<std::shared_ptr<polyfem::solver::Form>> &forms)
			: polyfem::solver::NLProblem(full_size, boundary_nodes, forms),
			  boundary_values_(boundary_values)
		{
		}

	protected:
		Eigen::MatrixXd boundary_values() const override { return boundary_values_; }

	private:
		const Eigen::MatrixXd boundary_values_;
	};

} // namespace polyfem::mesh