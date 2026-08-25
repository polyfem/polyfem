#pragma once

#include <polyfem/optimization/var2sims/VariableToSimulation.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace polyfem::solver
{
	/// @brief Maps to pressure boundary.
	///
	/// Maps one boundary solution to all boundaries in one boundary selection.
	/// Basically just the pressure version of dirichet boundary var2sim.
	///
	/// Expect optimiztion variable with layout:
	/// [ solution of boundary selection 1 at t0] [solution of boundary selection 2 at t0] ...
	/// [ solution of boundary selection 1 at t1] [solution of boundary selection 2 at t1] ...
	class PressureBoundaryVariableToSimulation : public VariableToSimulation
	{
	public:
		using StatePtrs = std::vector<std::shared_ptr<legacy::State>>;
		using DiffCachePtrs = std::vector<std::shared_ptr<DiffCache>>;

		/// @brief Construct ShapeVariableToSimulation.
		/// @param[in] states Shared ptr to all forward sim states.
		/// @param[in] diff_caches Shared ptr to all diff caches.
		/// @param[in] parametrizations Parametrizations.
		/// @param[in] active_boundary_ids Active pressure boundary ids. Empty implies all active.
		/// @param[in] active_time_slices_ Active time slices. Empty implies all active. Ignored for static problem.
		/// @throw std::runtime_error Throw if input is invalid.
		PressureBoundaryVariableToSimulation(
			StatePtrs states,
			DiffCachePtrs diff_caches,
			CompositeParametrization parametrizations,
			Eigen::VectorXi active_boundary_ids,
			Eigen::VectorXi active_time_slices);

		std::string name() const override;
		ParameterType parameter_type() const override;
		bool affect_state(const legacy::State &target) const override;
		void update(const Eigen::VectorXd &x) override;
		void update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const override;
		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		int inverse_dof() const override;
		Eigen::VectorXd inverse_eval() const override;
		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const override;

	private:
		bool is_transient_;
		int time_steps_;

		StatePtrs states_;
		DiffCachePtrs diff_caches_;
		CompositeParametrization parametrization_;

		Eigen::VectorXi active_boundary_ids_;
		Eigen::VectorXi active_time_slices_;

		int para_out_dof() const;
	};

} // namespace polyfem::solver
