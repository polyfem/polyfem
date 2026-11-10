#pragma once

#include <polyfem/optimization/var2sims/VariableToSimulation.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace polyfem::solver
{
	/// @brief Maps to initial solution and solution velocity.
	///
	/// Expect optimization variables with layout:
	/// [ initial solution of all dofs .... ] [ initail solution velocity of all dofs ... ]
	class InitialConditionVariableToSimulation : public VariableToSimulation
	{
	public:
		using VarFormPtrs = std::vector<std::shared_ptr<varform::DifferentiableVarForm>>;
		using DiffCachePtrs = std::vector<std::shared_ptr<DiffCache>>;

		/// @brief Construct InitialConditionVariableToSimulation.
		/// @param[in] varforms Shared ptr to all forward sim varforms.
		/// @param[in] diff_caches Shared ptr to all diff caches.
		/// @param[in] parametrizations Parametrizations.
		/// @param[in] active_dofs Active solution space dofs. Empty implies all active.
		/// @throw std::runtime_error Throw if input is invalid.
		InitialConditionVariableToSimulation(VarFormPtrs varforms,
											 DiffCachePtrs diff_caches,
											 CompositeParametrization parametrizations,
											 Eigen::VectorXi active_dofs);

		std::string name() const override;
		ParameterType parameter_type() const override;
		bool affects_varform(const varform::DifferentiableVarForm &target) const override;
		void update(const Eigen::VectorXd &x) override;
		void update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const override;
		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		int inverse_dof() const override;
		Eigen::VectorXd inverse_eval() const override;
		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const override;

	private:
		int dof_num_;
		VarFormPtrs varforms_;
		DiffCachePtrs diff_caches_;
		CompositeParametrization parametrization_;
		Eigen::VectorXi active_dofs_;

		int para_out_dof() const;
	};

} // namespace polyfem::solver
