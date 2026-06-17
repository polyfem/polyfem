#pragma once

#include <polyfem/optimization/var2sims/VariableToSimulation.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace polyfem::solver
{
	/// @brief Maps to dirichlet boundary.
	///
	/// Maps one boundary solution to all boundaries in one boundary selection.
	///
	/// Example 1:
	/// Suppose one boundary selection contains all vertices at x=0, this var2sim will map one
	/// solution (displacement xyz) to all vertices in the selection.
	///
	/// Example 2:
	/// Suppose there are two boundary selections -- the left and right surface of a cube.
	/// This var2sim will map two solution, displacement to all vertices in their respective
	/// boundary seletion.
	///
	/// Expect optimiztion variable with layout:
	/// [ solution of boundary selection 1 at t0] [solution of boundary selection 2 at t0] ...
	/// [ solution of boundary selection 1 at t1] [solution of boundary selection 2 at t1] ...
	class DirichletBoundaryVariableToSimulation : public VariableToSimulation
	{
	public:
		using StatePtrs = std::vector<std::shared_ptr<legacy::State>>;
		using DiffCachePtrs = std::vector<std::shared_ptr<DiffCache>>;

		/// @brief Construct DirichletBoundaryVariableToSimulation.
		/// @param[in] states Shared ptr to all forward sim states.
		/// @param[in] diff_caches Shared ptr to all diff caches.
		/// @param[in] parametrizations Parametrizations.
		/// @param[in] active_boundary_ids Active Dirichlet boundary ids. Empty implies all active.
		/// @param[in] active_time_slices Active time slices. Empty implies all active.
		/// @throw std::runtime_error Throw if input is invalid.
		DirichletBoundaryVariableToSimulation(
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
		/// boundary order in this var2sim -> component (dim) -> offset in state.boundary_nodes
		using BoundaryNodeMap = std::vector<std::vector<std::vector<int>>>;

		int dim_;
		int time_steps_;

		StatePtrs states_;
		DiffCachePtrs diff_caches_;
		CompositeParametrization parametrization_;

		Eigen::VectorXi active_boundary_ids_;
		Eigen::VectorXi active_time_slices_;

		/// boundary node map per state.
		std::vector<BoundaryNodeMap> boundary_node_maps_;

		int para_out_dof() const;
		void build_boundary_node_maps();
	};

} // namespace polyfem::solver
