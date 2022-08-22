#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	/// @brief Form of the elasticity potential and forces
	class ElasticForm : public Form
	{
	public:
		/// @brief Construct a new Elastic Form object
		/// @param state Reference to the simulation state
		ElasticForm(const State &state);

		/// @brief Compute the elastic potential value
		/// @param x Current solution
		/// @return Value of the elastic potential
		double value(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

		/// @brief Determine if a step from solution x0 to solution x1 is allowed
		/// @param x0 Current solution
		/// @param x1 Proposed next solution
		/// @return True if the step is allowed
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

	private:
		const State &state_;                         ///< Reference to the simulation state
		const assembler::AssemblerUtils &assembler_; ///< Reference to the assembler
		const std::string formulation_;              ///< Elasticity formulation name
		StiffnessMatrix cached_stiffness_;           ///< Cached stiffness matrix for linear elasticity
		utils::SpareMatrixCache mat_cache_;          ///< Matrix cache

		/// @brief Elasticity formulation name
		const std::string &formulation() const { return formulation_; }

		/// @brief Compute the stiffness matrix (cached)
		void compute_cached_stiffness();
	};
} // namespace polyfem::solver
