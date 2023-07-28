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
		Material,
		FrictionCoeff,
		DampingCoeff,
		InitialCondition,
		DirichletBC,
		MacroStrain
	};

	enum class SpatialIntegralType
	{
		volume,
		surface,
		vertex_sum
	};

	class AdjointTools
	{
	public:
		AdjointTools()
		{
		}

		static double integrate_objective(
			const State &state,
			const IntegrableFunctional &j,
			const Eigen::MatrixXd &solution,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			const int cur_step = 0);
		static void dJ_du_step(
			const State &state,
			const IntegrableFunctional &j,
			const Eigen::MatrixXd &solution,
			const std::set<int> &interested_ids,
			const SpatialIntegralType spatial_integral_type,
			const int cur_step,
			Eigen::VectorXd &term);
		static void dJ_macro_strain_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		static void compute_macro_strain_derivative_functional_term(
			const State &state,
			const Eigen::MatrixXd &solution,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			Eigen::VectorXd &term,
			const int cur_time_step);
		static void compute_shape_derivative_functional_term(
			const State &state,
			const Eigen::MatrixXd &solution,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			Eigen::VectorXd &term,
			const int cur_time_step);
		static void dJ_shape_static_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		static void dJ_shape_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_material_static_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		static void dJ_material_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_friction_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_damping_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_initial_condition_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_dirichlet_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);

		static Eigen::VectorXd map_primitive_to_node_order(
			const State &state,
			const Eigen::VectorXd &primitives);
		static Eigen::VectorXd map_node_to_primitive_order(
			const State &state,
			const Eigen::VectorXd &nodes);

		static Eigen::MatrixXd edge_normal_gradient(
			const Eigen::MatrixXd &V);
		static Eigen::MatrixXd face_normal_gradient(
			const Eigen::MatrixXd &V);

		static Eigen::MatrixXd edge_velocity_divergence(
			const Eigen::MatrixXd &V);
		static Eigen::MatrixXd face_velocity_divergence(
			const Eigen::MatrixXd &V);
	};
} // namespace polyfem::solver
