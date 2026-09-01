#pragma once

#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/parametrization/Parametrization.hpp>
#include <polyfem/optimization/var2sims/ParameterType.hpp>

#include <Eigen/Core>

#include <string>

namespace polyfem::solver
{
	/// @brief Maps optimization variables to forward simulation varform::DifferentiableVarForm variables.
	class VariableToSimulation
	{
	public:
		virtual ~VariableToSimulation() = default;

		virtual std::string name() const = 0;

		virtual ParameterType parameter_type() const = 0;

		/// @brief Return true if current var2sim maps to target varform.
		virtual bool affects_varform(const varform::DifferentiableVarForm &target) const = 0;

		/// @brief Update forward simulation varforms from optimization variables.
		/// @param[in] x Optimization variables.
		virtual void update(const Eigen::VectorXd &x) = 0;

		/// @brief Update varform variables from optimization variables.
		///
		/// Compared to update() this method update abstract varform variables
		/// instead of writing directly to varform.
		///
		/// @param[in] x Optimization variables.
		/// @param[out] state_variables Abstract varform variables update dst.
		virtual void update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const = 0;

		/// @brief Compute adjoint contribution of objective gradient.
		///
		/// See arXiv:2205.13643.
		///
		/// Let objective function be J and optimization variable be x.
		/// This method computes all adjoint related terms in dJ/dx. For
		/// static case (E.q. 11), it's the second term. For dynamic case
		/// (E.q. 15), it's everything except ∂qJ.
		///
		/// @param x[in] Optimization variables.
		/// @return Adjoint contribution of objective gradient.
		virtual Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const = 0;

		/// @brief Compute optimization variables dof.
		/// @return Optimization variables dof.
		/// @throw std::runtime_error Throw if not implemented.
		virtual int inverse_dof() const = 0;

		/// @brief Compute optimization variables from forward simulation varform::DifferentiableVarForm.
		/// @return Optimization variables.
		/// @throw std::runtime_error Throw if not implemented.
		virtual Eigen::VectorXd inverse_eval() const = 0;

		/// @brief Apply parametrization jacobian to compute the gradient w.r.t.
		/// to optimization variables.
		/// @param term Gradient w.r.t. to full inherent dof (Ex. all vertices for shape var2sim)
		/// @param x Optimization variables.
		virtual Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const = 0;
	};

} // namespace polyfem::solver
