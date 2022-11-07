#pragma once

#include <polyfem/State.hpp>
#include <polyfem/solver/Parameter.hpp>

namespace polyfem::solver
{
	class AdjointForm
	{
	public:
		AdjointForm()
		{
		}

		enum class SpatialIntegralType
		{
			VOLUME,
			SURFACE,
			VERTEX_SUM
		};

		static double value(
			const State &state,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			const std::string &transient_integral_type = "");

		static void gradient(
			State &state,
			const IntegrableFunctional &j,
			const std::string &param_name,
			Eigen::VectorXd &grad,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			const std::string &transient_integral_type = "");
		
		// assume adjoints are solved and stored in state
		static void compute_adjoint_term(
			const State &state,
			const std::string &param_name,
			Eigen::VectorXd &term);

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
		static double integrate_objective_transient(
			const State &state,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const SpatialIntegralType spatial_integral_type,
			const std::string &transient_integral_type);
		static void dJ_du_transient(
			const State &state,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const SpatialIntegralType spatial_integral_type,
			const std::string &transient_integral_type,
			Eigen::MatrixXd &terms);
		static void compute_topology_derivative_functional_term(
			const State &state,
			const Eigen::MatrixXd &solution,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			Eigen::VectorXd &term);
		static void compute_shape_derivative_functional_term(
			const State &state,
			const Eigen::MatrixXd &solution,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids, // either body id or surface id
			const SpatialIntegralType spatial_integral_type,
			Eigen::VectorXd &term,
			const int cur_time_step);
		static void dJ_topology_static(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const SpatialIntegralType spatial_integral_type,
			Eigen::VectorXd &one_form);
		static void dJ_topology_static_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		static void dJ_shape_static(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const SpatialIntegralType spatial_integral_type,
			Eigen::VectorXd &one_form);
		static void dJ_shape_static_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		static void dJ_shape_transient(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const SpatialIntegralType spatial_integral_type,
			const std::string &transient_integral_type,
			Eigen::VectorXd &one_form);
		static void dJ_shape_transient_adjoint_term(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_material_static(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		static void dJ_material_transient(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_friction_transient(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_damping_transient(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_initial_condition(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_dirichlet_transient(
			const State &state,
			const Eigen::MatrixXd &adjoint_nu,
			const Eigen::MatrixXd &adjoint_p,
			Eigen::VectorXd &one_form);
	};
} // namespace polyfem::solver
