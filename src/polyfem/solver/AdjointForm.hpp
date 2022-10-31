#pragma once

#include <polyfem/State.hpp>

namespace polyfem::solver
{
	class AdjointForm
	{
	public:
		AdjointForm(const std::string &type)
		{
			// TODO: build IntegrableFunctional j based on type
		}

		double value(
			const State &state,
			const std::set<int> &interested_ids, // either body id or surface id
			const bool is_volume_integral,
			const std::string &transient_integral_type = "")
		{
			if (state.problem->is_time_dependent())
				return integrate_objective(state, j, state.diff_cached[0].u, interested_ids, is_volume_integral);
			else
				return integrate_objective_transient(state, j, interested_ids, is_volume_integral, transient_integral_type);
		}

		void gradient(
			const State &state,
			const std::set<int> &interested_ids, // either body id or surface id
			const bool is_volume_integral,
			const std::string &param,
			Eigen::VectorXd &grad,
			const std::string &transient_integral_type = "")
		{
			if (param == "material") {}
			else if (param == "shape") {}
			else if (param == "friction") {}
			else if (param == "damping") {}
			else if (param == "initial") {}
			else if (param == "dirichlet") {}
			else
				log_and_throw_error("Unknown design parameter!");
		}

	private:
		IntegrableFunctional j;

	private:
		static double integrate_objective(
			const State &state, 
			const IntegrableFunctional &j, 
			const Eigen::MatrixXd &solution,
			const std::set<int> &interested_ids, // either body id or surface id
			const bool is_volume_integral,
			const int cur_step = 0);
		static double integrate_objective_transient(
			const State &state, 
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const bool is_volume_integral,
			const std::string &transient_integral_type);
		static void compute_shape_derivative_functional_term(
			const State &state,
			const Eigen::MatrixXd &solution, 
			const IntegrableFunctional &j, 
			const std::set<int> &interested_ids, // either body id or surface id 
			const bool is_volume_integral,
			Eigen::VectorXd &term, 
			const int cur_time_step);
		static void dJ_shape_static(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const bool is_volume_integral,
			Eigen::VectorXd &one_form);
		static void dJ_shape_transient(
			const State &state,
			const std::vector<Eigen::MatrixXd> &adjoint_nu,
			const std::vector<Eigen::MatrixXd> &adjoint_p,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const bool is_volume_integral,
			const std::string &transient_integral_type,
			Eigen::VectorXd &one_form);
		static void dJ_material_static(
			const State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &one_form);
		static void dJ_material_transient(
			const State &state,
			const std::vector<Eigen::MatrixXd> &adjoint_nu,
			const std::vector<Eigen::MatrixXd> &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_friction_transient(
			const State &state,
			const std::vector<Eigen::MatrixXd> &adjoint_nu,
			const std::vector<Eigen::MatrixXd> &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_damping_transient(
			const State &state,
			const std::vector<Eigen::MatrixXd> &adjoint_nu,
			const std::vector<Eigen::MatrixXd> &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_initial_condition(
			const State &state,
			const std::vector<Eigen::MatrixXd> &adjoint_nu,
			const std::vector<Eigen::MatrixXd> &adjoint_p,
			Eigen::VectorXd &one_form);
		static void dJ_dirichlet_transient(
			const State &state,
			const std::vector<Eigen::MatrixXd> &adjoint_nu,
			const std::vector<Eigen::MatrixXd> &adjoint_p,
			const IntegrableFunctional &j,
			const std::set<int> &interested_ids,
			const bool is_volume_integral,
			const std::string &transient_integral_type,
			Eigen::VectorXd &one_form);
	};
} // namespace polyfem::solver
