#pragma once

#include <polyfem/legacy/State.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/parametrization/Parametrization.hpp>
#include <polyfem/optimization/var2sims/ParameterType.hpp>

#include <Eigen/Core>

#include <string>

namespace polyfem::solver
{
	/// @brief Maps optimization variables to forward simulation legacy::State variables.
	class VariableToSimulation
	{
	public:
		virtual ~VariableToSimulation() = default;

		virtual std::string name() const = 0;

		virtual ParameterType parameter_type() const = 0;

		/// @brief Return true if current var2sim maps to target state.
		virtual bool affect_state(const legacy::State &target) const = 0;

		/// @brief Update forward simulation states from optimization variables.
		/// @param[in] x Optimization variables.
		virtual void update(const Eigen::VectorXd &x) = 0;

		/// @brief Update state variables from optimization variables.
		///
		/// Compared to update() this method update abstract state variables
		/// instead of writing directly to state.
		///
		/// @param[in] x Optimization variables.
		/// @param[out] state_variables Abstract state variables update dst.
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

		/// @brief Compute optimization variables from forward simulation legacy::State.
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
