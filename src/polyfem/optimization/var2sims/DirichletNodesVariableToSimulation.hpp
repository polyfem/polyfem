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
	/// Maps one boundary solution to one boundary. Only support static problem.
	///
	/// Example 1:
	/// Suppose one boundary selection contains 3 vertices at x=0, this var2sim will map 3
	/// solution (displacement xyz) to their respective vertices.
	class DirichletNodesVariableToSimulation : public VariableToSimulation
	{
	public:
		using StatePtrs = std::vector<std::shared_ptr<legacy::State>>;
		using DiffCachePtrs = std::vector<std::shared_ptr<DiffCache>>;

		/// @brief Construct DirichletNodesVariableToSimulation.
		/// @param[in] states Shared ptr to all forward sim states.
		/// @param[in] diff_caches Shared ptr to all diff caches.
		/// @param[in] parametrizations Parametrizations.
		/// @param[in] actice_dimensions Vector of active dimensions per mesh node. Empty implies all active.
		/// @param[in] active_geom_nodes Input vertex indices. Empty implies all nodal Dirichlet vertices.
		/// @throw std::runtime_error Throw if input is invalid.
		DirichletNodesVariableToSimulation(
			StatePtrs states,
			DiffCachePtrs diff_caches,
			CompositeParametrization parametrizations,
			Eigen::VectorXi active_geom_nodes);

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
		int dim_;
		int vertex_num_;

		StatePtrs states_;
		DiffCachePtrs diff_caches_;
		CompositeParametrization parametrization_;

		// Input vertex indices (0..n_vertices-1).
		Eigen::VectorXi active_geom_nodes_;

		int para_out_dof() const;
	};

} // namespace polyfem::solver
