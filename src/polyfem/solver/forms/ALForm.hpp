#pragma once

#include "Form.hpp"

#include <polyfem/assembler/RhsAssembler.hpp>

#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form of the augmented lagrangian
	class ALForm : public Form
	{
	public:
		/// @brief Construct a new AL Form object
		/// @param state Reference to the simulation state
		/// @param rhs_assembler Reference to the right hand side assembler
		/// @param t current time
		ALForm(const int ndof,
			   const std::vector<int> &boundary_nodes,
			   const std::vector<mesh::LocalBoundary> &local_boundary,
			   const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
			   const int n_boundary_samples,
			   const StiffnessMatrix &mass,
			   const assembler::RhsAssembler &rhs_assembler,
			   const mesh::Obstacle &obstacle,
			   const bool is_time_dependent,
			   const double t);

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
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		/// @brief Update time dependent quantities
		/// @param t New time
		/// @param x Solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override;

	private:
		const std::vector<int> &boundary_nodes_;
		const std::vector<mesh::LocalBoundary> &local_boundary_;
		const std::vector<mesh::LocalBoundary> &local_neumann_boundary_;
		const int n_boundary_samples_;

		const assembler::RhsAssembler &rhs_assembler_; ///< Reference to the RHS assembler
		const bool is_time_dependent_;

		StiffnessMatrix masked_lumped_mass_; ///< mass matrix masked by the AL dofs
		Eigen::MatrixXd target_x_;           ///< actually a vector with the same size as x with target nodal positions

		void update_target(const double t);
	};
} // namespace polyfem::solver
