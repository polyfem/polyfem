#pragma once

#include <polyfem/Common.hpp>
#include <set>

namespace polyfem
{
	class State;
	class IntegrableFunctional;
}

namespace polyfem::solver
{
	enum class ParameterType
	{
		Shape,
		LameParameter,
		FrictionCoefficient,
		DampingCoefficient,
		InitialCondition,
		DirichletBC,
		MacroStrain
	};

	enum class SpatialIntegralType
	{
		Volume,
		Surface,
		VertexSum
	};

	namespace AdjointTools
	{
		double integrate_objective(
			const State &state,
			const IntegrableFunctional &j,
			const Eigen::MatrixXd &solution,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			const int cur_step = 0);
		void dJ_du_step(
			const State &state,
			const IntegrableFunctional &j,
			const Eigen::MatrixXd &solution,
			const std::set<int> &interested_ids,
			const SpatialIntegralType spatial_integral_type,
			const int cur_step,
			Eigen::VectorXd &term);
		void compute_shape_derivative_functional_term(
			const State &state,
			const Eigen::MatrixXd &solution,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			Eigen::VectorXd &term,
			const int cur_time_step);
		void dJ_shape_static_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		void dJ_shape_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		void dJ_material_static_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		void dJ_material_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		void dJ_friction_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		void dJ_damping_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		void dJ_initial_condition_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		void dJ_dirichlet_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);

		Eigen::VectorXd map_primitive_to_node_order(
			const State &state,
			const Eigen::VectorXd &primitives);
		Eigen::VectorXd map_node_to_primitive_order(
			const State &state,
			const Eigen::VectorXd &nodes);

		Eigen::MatrixXd edge_normal_gradient(
			const Eigen::MatrixXd &V);
		Eigen::MatrixXd face_normal_gradient(
			const Eigen::MatrixXd &V);

		Eigen::MatrixXd edge_velocity_divergence(
			const Eigen::MatrixXd &V);
		Eigen::MatrixXd face_velocity_divergence(
			const Eigen::MatrixXd &V);
	};
} // namespace polyfem::solver
