#pragma once

#include "Form.hpp"

#include <polyfem/assembler/PressureAssembler.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/utils/Types.hpp>

#include <unordered_map>
#include <vector>

namespace polyfem::solver
{
	/// @brief Form representing body forces
	class PressureForm : public Form
	{
		friend class PressureForceDerivative;

	public:
		/// @brief Construct a new Body Form object
		/// @param state Reference to the simulation state
		/// @param pressure_assembler Reference to the pressure assembler
		PressureForm(const int ndof,
					 const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
					 const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity,
					 const std::vector<int> &dirichlet_nodes,
					 const QuadratureOrders &n_boundary_samples,
					 const assembler::PressureAssembler &pressure_assembler,
					 const bool is_time_dependent);

		std::string name() const override { return "pressure"; }

	protected:
		/// @brief Compute the value of the body force form
		/// @param x Current solution
		/// @return Value of the body force form
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Vector containing the current solution
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
		double t_;       ///< Current time
		const int ndof_; ///< Number of degrees of freedom

		const std::vector<mesh::LocalBoundary> &local_pressure_boundary_;
		const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity_;
		const std::vector<int> &dirichlet_nodes_;
		const QuadratureOrders n_boundary_samples_;

		const assembler::PressureAssembler &pressure_assembler_; ///< Reference to the pressure assembler
	};
} // namespace polyfem::solver
