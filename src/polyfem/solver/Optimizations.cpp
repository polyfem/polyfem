#include "OptimizationProblem.hpp"
#include "Optimizations.hpp"
#include "ShapeProblem.hpp"
#include <polyfem/assembler/RhsAssembler.hpp>
#include "TopologyOptimizationProblem.hpp"
#include "MaterialProblem.hpp"
#include "InitialConditionProblem.hpp"
#include "ControlProblem.hpp"
#include "GeneralOptimizationProblem.hpp"
#include "LBFGSBSolver.hpp"
#include "LBFGSSolver.hpp"
#include "BFGSSolver.hpp"
#include "MMASolver.hpp"
#include "GradientDescentSolver.hpp"
#include <polyfem/utils/CompositeSplineParam.hpp>

#include <map>

namespace polyfem
{
	double cross2(double x, double y)
	{
		x = abs(x);
		y = abs(y);
		if (x > y)
			std::swap(x, y);

		if (x < 0.1)
			return 0.05;
		return 0.95;
	}

	double cross3(double x, double y, double z)
	{
		x = abs(x);
		y = abs(y);
		z = abs(z);
		if (x > y)
			std::swap(x, y);
		if (y > z)
			std::swap(y, z);
		if (x > y)
			std::swap(x, y);

		if (y < 0.2)
			return 0.001;
		return 1;
	}

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_solver(const json &solver_params)
	{
		const std::string name = solver_params.contains("solver") ? solver_params["solver"].template get<std::string>() : "lbfgs";
		if (name == "GradientDescent" || name == "gradientdescent" || name == "gradient")
		{
			return std::make_shared<cppoptlib::GradientDescentSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "bfgs" || name == "BFGS" || name == "BFGS")
		{
			return std::make_shared<cppoptlib::BFGSSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "lbfgsb" || name == "LBFGSB" || name == "L-BFGS-B")
		{
			return std::make_shared<cppoptlib::LBFGSBSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "mma" || name == "MMA")
		{
			return std::make_shared<cppoptlib::MMASolver<ProblemType>>(
				solver_params);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

	double matrix_dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }

	std::shared_ptr<InitialConditionProblem> setup_initial_condition_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];

		std::shared_ptr<InitialConditionProblem> initial_problem = std::make_shared<InitialConditionProblem>(state, j);

		json initial_params;
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "initial")
			{
				initial_params = param;
				break;
			}
		}

		// fix certain object
		std::set<int> optimize_body_ids;
		if (initial_params.contains("volume_selection"))
		{
			for (int i : initial_params["volume_selection"])
				optimize_body_ids.insert(i);
		}
		else
			logger().info("No optimization body specified, optimize initial condition of every mesh...");

		// to get the initial velocity
		{
			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			json rhs_solver_params = state.args["solver"]["linear"];
			if (!rhs_solver_params.contains("Pardiso"))
				rhs_solver_params["Pardiso"] = {};
			rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

			state.solve_data.rhs_assembler = std::make_shared<assembler::RhsAssembler>(
				state.assembler, *state.mesh, state.obstacle, state.input_dirichlet,
				state.n_bases, state.problem->is_scalar() ? 1 : state.mesh->dimension(),
				state.bases, gbases, state.ass_vals_cache,
				state.formulation(), *state.problem,
				state.args["space"]["advanced"]["bc_method"],
				state.args["solver"]["linear"]["solver"], state.args["solver"]["linear"]["precond"], rhs_solver_params);

			assembler::RhsAssembler &rhs_assembler = *state.solve_data.rhs_assembler;
			rhs_assembler.initial_solution(state.initial_sol_update);
			rhs_assembler.initial_velocity(state.initial_vel_update);
		}
		auto initial_guess_vel = state.initial_vel_update, initial_guess_sol = state.initial_sol_update;

		const int dim = state.mesh->dimension();
		const int dof = state.n_bases;

		std::map<int, std::array<int, 2>> body_id_map; // from body_id to {node_id, index}
		int n = 0;
		for (int e = 0; e < state.bases.size(); e++)
		{
			const int body_id = state.mesh->get_body_id(e);
			if (!body_id_map.count(body_id) && (optimize_body_ids.count(body_id) || optimize_body_ids.size() == 0))
			{
				body_id_map[body_id] = {{state.bases[e].bases[0].global()[0].index, n}};
				n++;
			}
		}
		logger().info("{} objects found, each object has a constant initial velocity and position...", body_id_map.size());

		// by default optimize for initial velocity
		if (!initial_params.contains("restriction"))
			initial_params["restriction"] = "";

		if (initial_params["restriction"].get<std::string>() == "velocity")
		{
			initial_problem->x_to_param = [initial_guess_sol, initial_guess_vel, body_id_map, dim](const InitialConditionProblem::TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
				init_sol = initial_guess_sol;
				init_vel = initial_guess_vel;
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
							for (int d = 0; d < dim; d++)
								init_vel(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d);
				}
				std::cout << "initial velocity: " << std::setprecision(16) << x.transpose() << "\n";
			};

			initial_problem->param_to_x = [body_id_map, dim](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				for (auto i : body_id_map)
					for (int d = 0; d < dim; d++)
						x(i.second[1] * dim + d) = init_vel(i.second[0] * dim + d);
			};

			initial_problem->dparam_to_dx = [body_id_map, dim, dof](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				std::vector<bool> visited(dof, false);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
						{
							if (!visited[g.index])
								visited[g.index] = true;
							else
								continue;
							for (int d = 0; d < dim; d++)
								x(dim * body_id_map.at(body_id)[1] + d) += init_vel(g.index * dim + d);
						}
				}
			};
		}
		else if (initial_params["restriction"].get<std::string>() == "velocityXZ")
		{
			initial_problem->x_to_param = [initial_guess_sol, initial_guess_vel, body_id_map, dim](const InitialConditionProblem::TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
				init_sol = initial_guess_sol;
				init_vel = initial_guess_vel;
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
							for (int d = 0; d < dim; d++)
								init_vel(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d);
				}
				logger().debug("initial velocity: {}", x.transpose());
			};

			initial_problem->param_to_x = [body_id_map, dim](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				for (auto i : body_id_map)
					for (int d = 0; d < dim; d++)
						x(i.second[1] * dim + d) = init_vel(i.second[0] * dim + d);
				logger().debug("initial velocity: {}", x.transpose());
			};

			initial_problem->dparam_to_dx = [body_id_map, dim, dof](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				std::vector<bool> visited(dof, false);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
						{
							if (!visited[g.index])
								visited[g.index] = true;
							else
								continue;
							for (int d = 0; d < dim; d++)
							{
								if (d == 1)
									continue;
								x(dim * body_id_map.at(body_id)[1] + d) += init_vel(g.index * dim + d);
							}
						}
				}
			};
		}
		else if (initial_params["restriction"].get<std::string>() == "position")
		{
			initial_problem->x_to_param = [body_id_map, dim, dof](const InitialConditionProblem::TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
				init_sol.setZero(dof * dim, 1);
				init_vel.setZero(dof * dim, 1);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
							for (int d = 0; d < dim; d++)
								init_sol(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d);
				}
				logger().debug("initial solution: {}", x.transpose());
			};

			initial_problem->param_to_x = [body_id_map, dim](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				for (auto i : body_id_map)
				{
					for (int d = 0; d < dim; d++)
						x(i.second[1] * dim + d) = init_sol(i.second[0] * dim + d);
				}
				logger().debug("initial solution: {}", x.transpose());
			};

			initial_problem->dparam_to_dx = [body_id_map, dim, dof](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				std::vector<bool> visited(dof, false);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
						{
							if (!visited[g.index])
								visited[g.index] = true;
							else
								continue;
							for (int d = 0; d < dim; d++)
								x(dim * body_id_map.at(body_id)[1] + d) += init_sol(g.index * dim + d);
						}
				}
			};
		}
		else
		{
			initial_problem->x_to_param = [initial_guess_sol, initial_guess_vel, body_id_map, dim](const InitialConditionProblem::TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
				init_sol = initial_guess_sol;
				init_vel = initial_guess_vel;
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
							for (int d = 0; d < dim; d++)
							{
								init_sol(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d);
								init_vel(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d + dim * body_id_map.size());
							}
				}
				logger().debug("initial velocity: {}", x.tail(x.size() / 2).transpose());
				logger().debug("initial position: {}", x.head(x.size() / 2).transpose());
			};

			initial_problem->param_to_x = [body_id_map, dim](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size() * 2);
				for (auto i : body_id_map)
					for (int d = 0; d < dim; d++)
					{
						x(i.second[1] * dim + d) = init_sol(i.second[0] * dim + d);
						x(i.second[1] * dim + d + dim * body_id_map.size()) = init_vel(i.second[0] * dim + d);
					}
				logger().debug("initial velocity: {}", x.tail(x.size() / 2).transpose());
				logger().debug("initial position: {}", x.head(x.size() / 2).transpose());
			};

			initial_problem->dparam_to_dx = [body_id_map, dim, dof](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size() * 2);
				std::vector<bool> visited(dof, false);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
						{
							if (!visited[g.index])
								visited[g.index] = true;
							else
								continue;
							for (int d = 0; d < dim; d++)
							{
								x(dim * body_id_map.at(body_id)[1] + d) += init_sol(g.index * dim + d);
								x(dim * body_id_map.at(body_id)[1] + d + dim * body_id_map.size()) += init_vel(g.index * dim + d);
							}
						}
				}
			};
		}
		initial_problem->param_to_x(x_initial, state.initial_sol_update, state.initial_vel_update, state);

		return initial_problem;
	}

	std::shared_ptr<MaterialProblem> setup_material_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];
		json material_params;

		state.args["output"]["paraview"]["options"]["material"] = true;

		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "material")
			{
				material_params = param;
				break;
			}
		}

		std::shared_ptr<MaterialProblem> material_problem = std::make_shared<MaterialProblem>(state, j);

		// fix certain object
		std::set<int> optimize_body_ids;
		if (material_params.contains("volume_selection"))
		{
			for (int i : material_params["volume_selection"])
				optimize_body_ids.insert(i);
		}
		else
			logger().info("No optimization body specified, optimize material of every mesh...");

		const int dim = state.mesh->dimension();
		const int dof = state.bases.size();

		std::map<int, std::array<int, 2>> body_id_map; // from body_id to {elem_id, index}
		int n = 0;
		for (int e = 0; e < dof; e++)
		{
			const int body_id = state.mesh->get_body_id(e);
			if (!body_id_map.count(body_id) && (optimize_body_ids.count(body_id) || optimize_body_ids.size() == 0))
			{
				body_id_map[body_id] = {{e, n}};
				n++;
			}
		}

		// constraints on optimization
		if (material_params.contains("restriction"))
		{
			if (material_params["restriction"].get<std::string>() == "constant")
			{
				logger().info("{} objects found, each object has constant material parameters...", body_id_map.size());

				material_problem->x_to_param = [body_id_map, dof](const MaterialProblem::TVector &x, State &state) {
					auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
					auto cur_mus = state.assembler.lame_params().mu_mat_;

					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);

						cur_lambdas(e) = x(body_id_map.at(body_id)[1] * 2 + 0);
						cur_mus(e) = x(body_id_map.at(body_id)[1] * 2 + 1);
					}
					state.assembler.update_lame_params(cur_lambdas, cur_mus);
				};
				material_problem->param_to_x = [body_id_map, dof](MaterialProblem::TVector &x, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					x.setZero(2 * body_id_map.size());
					for (auto i : body_id_map)
					{
						x(i.second[1] * 2 + 0) = cur_lambdas(i.second[0]);
						x(i.second[1] * 2 + 1) = cur_mus(i.second[0]);
					}
					logger().debug("material: {}", x.transpose());
				};
				material_problem->dparam_to_dx = [body_id_map, dof](MaterialProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					dx.setZero(2 * body_id_map.size());
					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);
						dx(body_id_map.at(body_id)[1] * 2 + 0) += dparams(e);
						dx(body_id_map.at(body_id)[1] * 2 + 1) += dparams(e + dof);
					}
				};
			}
			else if (material_params["restriction"].get<std::string>() == "log")
			{
				material_problem->x_to_param = [dof](const MaterialProblem::TVector &x, State &state) {
					auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
					auto cur_mus = state.assembler.lame_params().mu_mat_;
					cur_mus = x.tail(dof).array().exp().matrix();
					cur_lambdas = x.head(dof).array().exp().matrix();
					state.assembler.update_lame_params(cur_lambdas, cur_mus);
				};

				material_problem->param_to_x = [dof](MaterialProblem::TVector &x, State &state) {
					x.resize(2 * dof);
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;

					x.tail(dof) = cur_mus.array().log().matrix();
					x.head(dof) = cur_lambdas.array().log().matrix();
				};

				material_problem->dparam_to_dx = [dof](MaterialProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					dx.resize(dof * 2);
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;

					dx = dparams.head(2 * dof);
					dx.tail(dof) = dx.tail(dof).array() * cur_mus.array();
					dx.head(dof) = dx.head(dof).array() * cur_lambdas.array();
				};
			}
			else if (material_params["restriction"].get<std::string>() == "constant_log")
			{
				logger().info("{} objects found, each object has constant material parameters...", body_id_map.size());

				material_problem->x_to_param = [body_id_map, dof](const MaterialProblem::TVector &x, State &state) {
					auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
					auto cur_mus = state.assembler.lame_params().mu_mat_;

					cur_lambdas.setConstant(dof, 1, std::exp(x(0)));
					cur_mus.setConstant(dof, 1, std::exp(x(1)));

					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);
						if (!body_id_map.count(body_id))
							continue;
						cur_lambdas(e) = std::exp(x(body_id_map.at(body_id)[1] * 2 + 0));
						cur_mus(e) = std::exp(x(body_id_map.at(body_id)[1] * 2 + 1));
					}
					Eigen::VectorXd x_display = Eigen::VectorXd::Zero(x.size());
					for (int i = 0; i < x.size(); i++)
						x_display(i) = std::exp(x(i));
					state.assembler.update_lame_params(cur_lambdas, cur_mus);
					logger().debug("material: {}", x_display.transpose());
				};
				material_problem->param_to_x = [body_id_map](MaterialProblem::TVector &x, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					x.setZero(2 * body_id_map.size());
					for (auto i : body_id_map)
					{
						x(i.second[1] * 2 + 0) = std::log(cur_lambdas(i.second[0]));
						x(i.second[1] * 2 + 1) = std::log(cur_mus(i.second[0]));
					}
					Eigen::VectorXd x_display = Eigen::VectorXd::Zero(x.size());
					for (int i = 0; i < x.size(); i++)
						x_display(i) = std::exp(x(i));
					logger().debug("material: {}", x_display.transpose());
				};
				material_problem->dparam_to_dx = [body_id_map, dof](MaterialProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					dx.setZero(2 * body_id_map.size());
					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);
						if (!body_id_map.count(body_id))
							continue;
						dx(body_id_map.at(body_id)[1] * 2 + 0) += dparams(e) * cur_lambdas(e);
						dx(body_id_map.at(body_id)[1] * 2 + 1) += dparams(e + dof) * cur_mus(e);
					}
				};
			}
			else if (material_params["restriction"].get<std::string>() == "constant_log_friction")
			{
				logger().info("{} objects found, each object has constant material parameters...", body_id_map.size());

				material_problem->x_to_param = [body_id_map, dof](const MaterialProblem::TVector &x, State &state) {
					auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
					auto cur_mus = state.assembler.lame_params().mu_mat_;

					cur_lambdas.setConstant(dof, 1, std::exp(x(0)));
					cur_mus.setConstant(dof, 1, std::exp(x(1)));

					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);

						cur_lambdas(e) = std::exp(x(body_id_map.at(body_id)[1] * 2 + 0));
						cur_mus(e) = std::exp(x(body_id_map.at(body_id)[1] * 2 + 1));
					}
					state.assembler.update_lame_params(cur_lambdas, cur_mus);

					state.args["contact"]["friction_coefficient"] = std::exp(x(x.size() - 1));

					Eigen::VectorXd x_display = Eigen::VectorXd::Zero(x.size() - 1);
					for (int i = 0; i < x_display.size(); i++)
						x_display(i) = std::exp(x(i));
					logger().debug("material: {}", x_display.transpose());
					logger().debug("friction coeff = {}", state.args["contact"]["friction_coefficient"].get<double>());
				};
				material_problem->param_to_x = [body_id_map](MaterialProblem::TVector &x, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					x.setZero(2 * body_id_map.size() + 1);
					for (auto i : body_id_map)
					{
						x(i.second[1] * 2 + 0) = std::log(cur_lambdas(i.second[0]));
						x(i.second[1] * 2 + 1) = std::log(cur_mus(i.second[0]));
					}
					x(x.size() - 1) = std::log(state.args["contact"]["friction_coefficient"].get<double>());

					Eigen::VectorXd x_display = Eigen::VectorXd::Zero(x.size() - 1);
					for (int i = 0; i < x_display.size(); i++)
						x_display(i) = std::exp(x(i));
					logger().debug("material: {}", x_display.transpose());
					logger().debug("friction coeff = {}", state.args["contact"]["friction_coefficient"].get<double>());
				};
				material_problem->dparam_to_dx = [body_id_map, dof](MaterialProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					dx.setZero(2 * body_id_map.size() + 1);
					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);
						dx(body_id_map.at(body_id)[1] * 2 + 0) += dparams(e) * cur_lambdas(e);
						dx(body_id_map.at(body_id)[1] * 2 + 1) += dparams(e + dof) * cur_mus(e);
					}
					dx(dx.size() - 1) = dparams(2 * dof) * state.args["contact"]["friction_coefficient"].get<double>();
				};
			}
			else if (material_params["restriction"].get<std::string>() == "constant_log_friction_damping")
			{
				logger().info("{} objects found, each object has constant material parameters...", body_id_map.size());

				material_problem->x_to_param = [body_id_map, dof](const MaterialProblem::TVector &x, State &state) {
					auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
					auto cur_mus = state.assembler.lame_params().mu_mat_;

					cur_lambdas.setConstant(dof, 1, std::exp(x(0)));
					cur_mus.setConstant(dof, 1, std::exp(x(1)));

					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);

						cur_lambdas(e) = std::exp(x(body_id_map.at(body_id)[1] * 2 + 0));
						cur_mus(e) = std::exp(x(body_id_map.at(body_id)[1] * 2 + 1));
					}
					state.assembler.update_lame_params(cur_lambdas, cur_mus);

					state.args["contact"]["friction_coefficient"] = std::exp(x(x.size() - 3));
					state.damping_assembler.local_assembler().set_params(std::exp(x(x.size() - 2)), std::exp(x(x.size() - 1)));

					Eigen::VectorXd x_display = Eigen::VectorXd::Zero(x.size() - 3);
					for (int i = 0; i < x_display.size(); i++)
						x_display(i) = std::exp(x(i));
					logger().debug("material: {}", x_display.transpose());
					logger().debug("friction coeff = {}", state.args["contact"]["friction_coefficient"].get<double>());
					logger().debug("psi = {}, phi = {}", state.damping_assembler.local_assembler().get_psi(), state.damping_assembler.local_assembler().get_phi());
				};
				material_problem->param_to_x = [body_id_map](MaterialProblem::TVector &x, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					x.setZero(2 * body_id_map.size() + 3);
					for (auto i : body_id_map)
					{
						x(i.second[1] * 2 + 0) = std::log(cur_lambdas(i.second[0]));
						x(i.second[1] * 2 + 1) = std::log(cur_mus(i.second[0]));
					}
					x(x.size() - 3) = std::log(state.args["contact"]["friction_coefficient"].get<double>());
					x(x.size() - 2) = std::log(state.damping_assembler.local_assembler().get_psi());
					x(x.size() - 1) = std::log(state.damping_assembler.local_assembler().get_phi());

					Eigen::VectorXd x_display = Eigen::VectorXd::Zero(x.size() - 3);
					for (int i = 0; i < x_display.size(); i++)
						x_display(i) = std::exp(x(i));
					logger().debug("material: {}", x_display.transpose());
					logger().debug("friction coeff = {}", state.args["contact"]["friction_coefficient"].get<double>());
					logger().debug("psi = {}, phi = {}", state.damping_assembler.local_assembler().get_psi(), state.damping_assembler.local_assembler().get_phi());
				};
				material_problem->dparam_to_dx = [body_id_map, dof](MaterialProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					dx.setZero(2 * body_id_map.size() + 3);
					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);
						dx(body_id_map.at(body_id)[1] * 2 + 0) += dparams(e) * cur_lambdas(e);
						dx(body_id_map.at(body_id)[1] * 2 + 1) += dparams(e + dof) * cur_mus(e);
					}
					dx.tail(3) = dparams.tail(3);
					dx(dx.size() - 3) = dparams(dparams.size() - 3) * state.args["contact"]["friction_coefficient"].get<double>();
					dx(dx.size() - 2) = dparams(dparams.size() - 2) * state.damping_assembler.local_assembler().get_psi();
					dx(dx.size() - 1) = dparams(dparams.size() - 1) * state.damping_assembler.local_assembler().get_phi();
				};
			}
			else if (material_params["restriction"].get<std::string>() == "friction_damping")
			{
				material_problem->x_to_param = [](const MaterialProblem::TVector &x, State &state) {
					state.args["contact"]["friction_coefficient"] = x(x.size() - 3);
					state.damping_assembler.local_assembler().set_params(x(x.size() - 2), x(x.size() - 1));

					logger().debug("friction coeff = {}", state.args["contact"]["friction_coefficient"].get<double>());
					logger().debug("psi = {}, phi = {}", state.damping_assembler.local_assembler().get_psi(), state.damping_assembler.local_assembler().get_phi());
				};
				material_problem->param_to_x = [](MaterialProblem::TVector &x, State &state) {
					x.setZero(3);

					x(x.size() - 3) = state.args["contact"]["friction_coefficient"].get<double>();
					x(x.size() - 2) = state.damping_assembler.local_assembler().get_psi();
					x(x.size() - 1) = state.damping_assembler.local_assembler().get_phi();

					logger().debug("friction coeff = {}", state.args["contact"]["friction_coefficient"].get<double>());
					logger().debug("psi = {}, phi = {}", state.damping_assembler.local_assembler().get_psi(), state.damping_assembler.local_assembler().get_phi());
				};
				material_problem->dparam_to_dx = [](MaterialProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					dx.setZero(3);

					dx.tail(3) = dparams.tail(3);

					logger().debug("opt grad: {}", dx.transpose());
				};
			}
			else if (material_params["restriction"].get<std::string>() == "friction")
			{
				material_problem->x_to_param = [](const MaterialProblem::TVector &x, State &state) {
					state.args["contact"]["friction_coefficient"] = x(0);

					logger().debug("friction coeff = {}", state.args["contact"]["friction_coefficient"].get<double>());
				};
				material_problem->param_to_x = [](MaterialProblem::TVector &x, State &state) {
					x.setZero(1);

					x(0) = state.args["contact"]["friction_coefficient"].get<double>();

					logger().debug("friction coeff = {}", state.args["contact"]["friction_coefficient"].get<double>());
				};
				material_problem->dparam_to_dx = [](MaterialProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					dx.setZero(1);

					dx(0) = dparams(dparams.size() - 3);

					logger().debug("opt grad: {}", dx.transpose());
				};
			}
			else if (material_params["restriction"].get<std::string>() == "damping")
			{
				material_problem->x_to_param = [](const MaterialProblem::TVector &x, State &state) {
					state.damping_assembler.local_assembler().set_params(x(x.size() - 2), x(x.size() - 1));
					logger().debug("psi = {}, phi = {}", state.damping_assembler.local_assembler().get_psi(), state.damping_assembler.local_assembler().get_phi());
				};
				material_problem->param_to_x = [](MaterialProblem::TVector &x, State &state) {
					x.setZero(2);

					x(x.size() - 2) = state.damping_assembler.local_assembler().get_psi();
					x(x.size() - 1) = state.damping_assembler.local_assembler().get_phi();

					logger().debug("psi = {}, phi = {}", state.damping_assembler.local_assembler().get_psi(), state.damping_assembler.local_assembler().get_phi());
				};
				material_problem->dparam_to_dx = [](MaterialProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					dx.setZero(2);

					dx.tail(2) = dparams.tail(2);

					logger().debug("opt grad: {}", dx.transpose());
				};
			}
		}

		material_problem->param_to_x(x_initial, state);

		return material_problem;
	}

	std::shared_ptr<ShapeProblem> setup_shape_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{

		const auto &opt_params = state.args["optimization"];

		std::shared_ptr<ShapeProblem> shape_problem = std::make_shared<ShapeProblem>(state, j);

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state.get_vf(V, F);

		const auto &shape_params = shape_problem->get_shape_params();
		if (shape_params.contains("restriction"))
		{
			if (shape_params["restriction"].get<std::string>() == "cubic_hermite_spline" and shape_params.contains("spline_specification"))
			{
				// Assume there is one spline with id 10.
				const auto &spline_params = shape_params["spline_specification"];
				std::map<int, Eigen::MatrixXd> control_point, tangent;
				int sampling;
				bool couple_tangents; // couple the direction and magnitude of adjacent tangents
				int opt_dof = 0;
				int dim = state.mesh->dimension();
				assert(dim == 2);
				for (const auto &spline : spline_params)
				{
					const int boundary_id = spline["id"].get<int>();
					auto control_points = spline["control_point"];
					auto tangents = spline["tangent"];
					sampling = spline["sampling"].get<int>();
					Eigen::MatrixXd c(control_points.size(), dim), t(2 * control_points.size() - 2, dim);
					if (control_points.size() == tangents.size())
					{
						couple_tangents = true;
						for (int i = 0; i < control_points.size(); ++i)
						{
							assert(control_points[i].size() == dim);
							assert(tangents[i].size() == dim);
							for (int k = 0; k < dim; ++k)
							{
								c(i, k) = control_points[i][k];
								for (int j = 0; j < 2; ++j)
								{
									if (i != 0)
										t(2 * i - 1, k) = tangents[i][k];
									if (i != (control_points.size() - 1))
										t(2 * i, k) = tangents[i][k];
								}
							}
						}
					}
					else if ((2 * control_points.size() - 2) == tangents.size())
					{
						couple_tangents = false;
						for (int i = 0; i < control_points.size(); ++i)
						{
							assert(control_points[i].size() == dim);
							for (int k = 0; k < dim; ++k)
								c(i, k) = control_points[i][k];
						}
						for (int i = 0; i < tangents.size(); ++i)
						{
							assert(tangents[i].size() == dim);
							for (int k = 0; k < dim; ++k)
								t(i, k) = tangents[i][k];
						}
					}
					else
					{
						logger().error("The number of tangents must be either equal to (or twice of) number of control points.");
					}

					control_point.insert({boundary_id, c});
					tangent.insert({boundary_id, t});
					opt_dof += 2 * (c.rows() - 2);
					opt_dof += 2 * t.rows();
					logger().trace("Given tangents are: {}", t);
				}
				CompositeSplineParam spline_param(control_point, tangent, shape_problem->optimization_boundary_to_node, V, sampling);
				shape_problem->param_to_x = [spline_param, opt_dof, dim](ShapeProblem::TVector &x, const Eigen::MatrixXd &V) {
					std::map<int, Eigen::MatrixXd> control_point, tangent;
					spline_param.get_parameters(V, control_point, tangent);
					x.setZero(opt_dof);
					int index = 0;
					for (const auto &kv : control_point)
					{
						for (int i = 0; i < kv.second.rows(); ++i)
						{
							if (i == 0 || i == (kv.second.rows() - 1))
								continue;
							x.segment(index, dim) = kv.second.row(i);
							index += dim;
						}
					}
					for (const auto &kv : tangent)
					{
						for (int i = 0; i < kv.second.rows(); ++i)
						{
							x.segment(index, dim) = kv.second.row(i);
							index += dim;
						}
					}
					assert(index == x.size());
				};
				shape_problem->x_to_param = [control_point, tangent, spline_param](const ShapeProblem::TVector &x, const Eigen::MatrixXd &V_prev, Eigen::MatrixXd &V_) {
					std::map<int, Eigen::MatrixXd> new_control_point, new_tangent;
					int index = 0;
					for (const auto &kv : control_point)
					{
						Eigen::MatrixXd control_point_matrix(kv.second.rows(), kv.second.cols());
						for (int i = 0; i < kv.second.rows(); ++i)
						{
							if (i == 0 || i == (kv.second.rows() - 1))
								control_point_matrix.row(i) = kv.second.row(i);
							else
							{
								control_point_matrix.row(i) = x.segment(index, kv.second.cols());
								index += kv.second.cols();
							}
						}
						new_control_point[kv.first] = control_point_matrix;
					}
					for (const auto &kv : tangent)
					{
						Eigen::MatrixXd tangent_matrix(kv.second.rows(), kv.second.cols());
						for (int i = 0; i < kv.second.rows(); ++i)
						{
							tangent_matrix.row(i) = x.segment(index, kv.second.cols());
							index += kv.second.cols();
						}
						new_tangent[kv.first] = tangent_matrix;
					}
					spline_param.reparametrize(new_control_point, new_tangent, V_prev, V_);
				};
				shape_problem->dparam_to_dx = [control_point, spline_param, opt_dof, dim, couple_tangents](ShapeProblem::TVector &grad_x, const ShapeProblem::TVector &grad_v) {
					grad_x.setZero(opt_dof);
					int index = 0;
					for (const auto &kv : control_point)
					{
						Eigen::VectorXd grad_control_point, grad_tangent;
						spline_param.derivative_wrt_params(grad_v, kv.first, couple_tangents, grad_control_point, grad_tangent);
						grad_x.segment(index, grad_control_point.rows() - 2 * dim) = grad_control_point.segment(dim, grad_control_point.rows() - 2 * dim);
						index += grad_control_point.rows() - 2 * dim;
						grad_x.segment(index, grad_tangent.rows()) = grad_tangent;
						index += grad_tangent.rows();
					}
				};
			}
		}

		shape_problem->param_to_x(x_initial, V);
		shape_problem->set_optimization_dim(x_initial.size());

		return shape_problem;
	}

	std::shared_ptr<TopologyOptimizationProblem> setup_topology_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];

		std::shared_ptr<TopologyOptimizationProblem> top_opt = std::make_shared<TopologyOptimizationProblem>(state, j);

		Eigen::MatrixXd density_mat = state.assembler.lame_params().density_mat_;
		if (density_mat.size() != state.bases.size())
			density_mat.setZero(state.bases.size(), 1);
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "topology")
			{
				if (param.contains("initial"))
					density_mat.setConstant(param["initial"]);
				else
				{
					Eigen::MatrixXd barycenters;
					if (state.mesh->is_volume())
					{
						state.mesh->cell_barycenters(barycenters);
						for (int e = 0; e < state.bases.size(); e++)
						{
							density_mat(e) = cross3(barycenters(e, 0), barycenters(e, 1), barycenters(e, 2));
						}
					}
					else
					{
						state.mesh->face_barycenters(barycenters);
						for (int e = 0; e < state.bases.size(); e++)
						{
							density_mat(e) = cross2(barycenters(e, 0), barycenters(e, 1));
						}
					}
					// density_mat.setOnes();
				}

				if (param.contains("power"))
					state.assembler.update_lame_params_density(top_opt->apply_filter(density_mat), param["power"]);
				else
					state.assembler.update_lame_params_density(top_opt->apply_filter(density_mat));
				break;
			}
		}

		x_initial = density_mat;

		return top_opt;
	}

	std::shared_ptr<ControlProblem> setup_control_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];
		const auto &opt_nl_params = state.args["solver"]["optimization_nonlinear"];

		std::shared_ptr<ControlProblem> control_problem = std::make_shared<ControlProblem>(state, j);
		std::shared_ptr<cppoptlib::NonlinearSolver<ControlProblem>> nlsolver = make_nl_solver<ControlProblem>(opt_nl_params);
		nlsolver->setLineSearch(opt_nl_params["line_search"]["method"]);

		Eigen::VectorXd x;
		x.setZero(control_problem->get_optimize_boundary_ids_to_position().size() * state.mesh->dimension() * state.args["time"]["time_steps"].get<int>());
		for (int i = 0; i < state.args["boundary_conditions"]["dirichlet_boundary"].size(); ++i)
			if (control_problem->get_optimize_boundary_ids_to_position().count(state.args["boundary_conditions"]["dirichlet_boundary"][i]["id"].get<int>()) != 0)
			{
				int position = control_problem->get_optimize_boundary_ids_to_position().at(state.args["boundary_conditions"]["dirichlet_boundary"][i]["id"].get<int>());
				for (int k = 0; k < state.mesh->dimension(); ++k)
					for (int t = 0; t < state.args["time"]["time_steps"]; ++t)
						x(t * control_problem->get_optimize_boundary_ids_to_position().size() * state.mesh->dimension() + position * state.mesh->dimension() + k) = state.args["boundary_conditions"]["dirichlet_boundary"][i]["value"][k][t].get<double>();
			}
		// logger().info("Starting x: {}", x);

		x_initial = x;

		return control_problem;
	}

	void initial_condition_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		const auto &opt_nl_params = state.args["solver"]["optimization_nonlinear"];
		std::shared_ptr<cppoptlib::NonlinearSolver<InitialConditionProblem>> nlsolver = make_nl_solver<InitialConditionProblem>(opt_nl_params);
		nlsolver->setLineSearch(opt_nl_params["line_search"]["method"]);

		Eigen::VectorXd x;
		auto initial_problem = setup_initial_condition_optimization(state, j, x);

		nlsolver->minimize(*initial_problem, x);

		json solver_info;
		nlsolver->getInfo(solver_info);
		std::cout << solver_info << std::endl;
	}

	void material_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		const auto &opt_nl_params = state.args["solver"]["optimization_nonlinear"];
		std::shared_ptr<cppoptlib::NonlinearSolver<MaterialProblem>> nlsolver = make_nl_solver<MaterialProblem>(opt_nl_params);
		nlsolver->setLineSearch(opt_nl_params["line_search"]["method"]);

		Eigen::VectorXd x;
		auto material_problem = setup_material_optimization(state, j, x);

		nlsolver->minimize(*material_problem, x);

		json solver_info;
		nlsolver->getInfo(solver_info);
		std::cout << solver_info << std::endl;
	}

	void shape_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		const auto &opt_nl_params = state.args["solver"]["optimization_nonlinear"];
		std::shared_ptr<cppoptlib::NonlinearSolver<ShapeProblem>> nlsolver = make_nl_solver<ShapeProblem>(opt_nl_params);
		nlsolver->setLineSearch(opt_nl_params["line_search"]["method"]);

		Eigen::VectorXd x;
		auto shape_problem = setup_shape_optimization(state, j, x);

		nlsolver->minimize(*shape_problem, x);

		json solver_info;
		nlsolver->getInfo(solver_info);
		std::cout << solver_info << std::endl;
	}

	void topology_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		const auto &opt_nl_params = state.args["solver"]["optimization_nonlinear"];
		std::shared_ptr<cppoptlib::NonlinearSolver<TopologyOptimizationProblem>> nlsolver = make_nl_solver<TopologyOptimizationProblem>(opt_nl_params);
		nlsolver->setLineSearch(opt_nl_params["line_search"]["method"]);

		Eigen::VectorXd x;
		auto top_opt = setup_topology_optimization(state, j, x);

		nlsolver->minimize(*top_opt, x);

		json solver_info;
		nlsolver->getInfo(solver_info);
		std::cout << solver_info << std::endl;
	}

	void control_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		const auto &opt_nl_params = state.args["solver"]["optimization_nonlinear"];

		std::shared_ptr<cppoptlib::NonlinearSolver<ControlProblem>> nlsolver = make_nl_solver<ControlProblem>(opt_nl_params);
		nlsolver->setLineSearch(opt_nl_params["line_search"]["method"]);

		Eigen::VectorXd x;
		auto control_problem = setup_control_optimization(state, j, x);
		nlsolver->minimize(*control_problem, x);

		json solver_info;
		nlsolver->getInfo(solver_info);
		std::cout << solver_info << std::endl;
	}

	std::shared_ptr<GeneralOptimizationProblem> setup_general_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];

		std::vector<std::shared_ptr<OptimizationProblem>> problems;
		std::vector<Eigen::VectorXd> x_initial_list;
		int x_initial_size = 0;
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "shape")
			{
				Eigen::VectorXd tmp;
				problems.push_back(setup_shape_optimization(state, j, tmp));
				x_initial_size += tmp.size();
				x_initial_list.push_back(tmp);
			}
			else if (param["type"] == "control")
			{
				Eigen::VectorXd tmp;
				problems.push_back(setup_control_optimization(state, j, tmp));
				x_initial_size += tmp.size();
				x_initial_list.push_back(tmp);
			}
			else
			{
				logger().error("General optimization with {} not currently supported.", param["type"]);
			}
		}

		x_initial.resize(x_initial_size);
		int count = 0;
		for (const auto &x : x_initial_list)
		{
			x_initial.segment(count, x.size()) = x;
			count += x.size();
			logger().trace("Size of initial guess is {}", x.size());
		}

		std::shared_ptr<GeneralOptimizationProblem>
			general_optimization_problem = std::make_shared<GeneralOptimizationProblem>(state, problems, j);

		return general_optimization_problem;
	}

	void general_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		const auto &opt_nl_params = state.args["solver"]["optimization_nonlinear"];
		std::shared_ptr<cppoptlib::NonlinearSolver<GeneralOptimizationProblem>> nlsolver = make_nl_solver<GeneralOptimizationProblem>(opt_nl_params);
		nlsolver->setLineSearch(opt_nl_params["line_search"]["method"]);

		Eigen::VectorXd x;
		auto general_optimization_problem = setup_general_optimization(state, j, x);
		nlsolver->minimize(*general_optimization_problem, x);

		json solver_info;
		nlsolver->getInfo(solver_info);
		std::cout << solver_info << std::endl;
	}
} // namespace polyfem