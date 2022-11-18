#include "Optimizations.hpp"

#include "ShapeProblem.hpp"
#include "ElasticProblem.hpp"
#include "InitialConditionProblem.hpp"
#include "ControlProblem.hpp"
#include "FrictionProblem.hpp"
#include "DampingProblem.hpp"
#include "GeneralOptimizationProblem.hpp"

#include <polyfem/utils/CompositeSplineParam.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>

#include <map>

namespace polyfem::solver
{
	namespace {
		bool load_json(const std::string &json_file, json &out)
		{
			std::ifstream file(json_file);

			if (!file.is_open())
				return false;

			file >> out;

			out["root_path"] = json_file;

			return true;
		}

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

		double matrix_dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }
	}

	std::shared_ptr<OptimizationProblem> setup_initial_condition_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
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
		if (initial_params["volume_selection"].size() > 0)
		{
			for (int i : initial_params["volume_selection"])
				optimize_body_ids.insert(i);
		}
		else
			logger().info("No optimization body specified, optimize initial condition of every mesh...");

		// to get the initial velocity
		{
			state.solve_data.rhs_assembler = state.build_rhs_assembler();
			state.solve_data.rhs_assembler->initial_solution(state.initial_sol_update);
			state.solve_data.rhs_assembler->initial_velocity(state.initial_vel_update);
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
		initial_problem->set_optimization_dim(x_initial.size());

		return initial_problem;
	}

	std::shared_ptr<OptimizationProblem> setup_material_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
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

		std::shared_ptr<ElasticProblem> material_problem = std::make_shared<ElasticProblem>(state, j);

		// fix certain object
		std::set<int> optimize_body_ids;
		if (material_params["volume_selection"].size() > 0)
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
		{
			if (material_params["restriction"].get<std::string>() == "constant")
			{
				logger().info("{} objects found, each object has constant material parameters...", body_id_map.size());

				material_problem->x_to_param = [body_id_map, dof](const ElasticProblem::TVector &x, State &state) {
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
				material_problem->param_to_x = [body_id_map, dof](ElasticProblem::TVector &x, State &state) {
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
				material_problem->dparam_to_dx = [body_id_map, dof](ElasticProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
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
				material_problem->x_to_param = [dof](const ElasticProblem::TVector &x, State &state) {
					auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
					auto cur_mus = state.assembler.lame_params().mu_mat_;
					cur_mus = x.tail(dof).array().exp().matrix();
					cur_lambdas = x.head(dof).array().exp().matrix();
					state.assembler.update_lame_params(cur_lambdas, cur_mus);
				};

				material_problem->param_to_x = [dof](ElasticProblem::TVector &x, State &state) {
					x.resize(2 * dof);
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;

					x.tail(dof) = cur_mus.array().log().matrix();
					x.head(dof) = cur_lambdas.array().log().matrix();
				};

				material_problem->dparam_to_dx = [dof](ElasticProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					dx.resize(dof * 2);
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;

					dx = dparams.head(2 * dof);
					dx.tail(dof) = dx.tail(dof).array() * cur_mus.array();
					dx.head(dof) = dx.head(dof).array() * cur_lambdas.array();
				};
			}
			else if (material_params["restriction"].get<std::string>() == "E_nu")
			{
				material_problem->design_variable_name = "E_nu";
				material_problem->x_to_param = [dof](const ElasticProblem::TVector &x, State &state) {
					auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
					auto cur_mus = state.assembler.lame_params().mu_mat_;

					for (int e = 0; e < dof; e++)
					{
						const double E = x(e * 2 + 0);
						const double nu = x(e * 2 + 1);

						cur_lambdas(e) = convert_to_lambda(state.mesh->is_volume(), E, nu);
						cur_mus(e) = convert_to_mu(E, nu);
					}
					state.assembler.update_lame_params(cur_lambdas, cur_mus);
				};
				material_problem->param_to_x = [dof, dim](ElasticProblem::TVector &x, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					x.setZero(dof * 2);
					for (int e = 0; e < dof; e++)
					{
						x(e * 2 + 0) = convert_to_E(state.mesh->is_volume(), cur_lambdas(e), cur_mus(e));
						x(e * 2 + 1) = convert_to_nu(state.mesh->is_volume(), cur_lambdas(e), cur_mus(e));
					}
				};
				material_problem->dparam_to_dx = [dof, dim](ElasticProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					dx.setZero(dof * 2);

					for (int e = 0; e < dof; e++)
					{
						const double E = convert_to_E(state.mesh->is_volume(), cur_lambdas(e), cur_mus(e));
						const double nu = convert_to_nu(state.mesh->is_volume(), cur_lambdas(e), cur_mus(e));
						// const double dlambda_dnu = E * (1 + 2 * nu * nu) / pow(2 * nu * nu + nu - 1, 2);
						// const double dmu_dnu = -E / 2 / pow(1 + nu, 2);
						// const double dlambda_dE = nu / (2 * nu * nu + nu - 1);
						// const double dmu_dE = 1 / 2 / (1 + nu);

						Eigen::Matrix2d jacobian = d_lambda_mu_d_E_nu(state.mesh->is_volume(), E, nu);

						dx(e * 2 + 0) += dparams(e) * jacobian(0, 0) + dparams(e + dof) * jacobian(1, 0);
						dx(e * 2 + 1) += dparams(e) * jacobian(0, 1) + dparams(e + dof) * jacobian(1, 1);
					}
				};
			}
			else if (material_params["restriction"].get<std::string>() == "constant_E_nu")
			{
				logger().info("{} objects found, each object has constant material parameter nu...", body_id_map.size());

				material_problem->design_variable_name = "E_nu";
				material_problem->x_to_param = [body_id_map, dof](const ElasticProblem::TVector &x, State &state) {
					auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
					auto cur_mus = state.assembler.lame_params().mu_mat_;

					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);

						if (!body_id_map.count(body_id))
							continue;

						const double E = x(body_id_map.at(body_id)[1] * 2 + 0);
						const double nu = x(body_id_map.at(body_id)[1] * 2 + 1);

						cur_lambdas(e) = convert_to_lambda(state.mesh->is_volume(), E, nu);
						cur_mus(e) = convert_to_mu(E, nu);
					}
					state.assembler.update_lame_params(cur_lambdas, cur_mus);
					logger().debug("material E nu: {}", x.transpose());
				};
				material_problem->param_to_x = [body_id_map, dof](ElasticProblem::TVector &x, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					x.setZero(body_id_map.size() * 2);
					for (auto i : body_id_map)
					{
						x(i.second[1] * 2 + 0) = convert_to_E(state.mesh->is_volume(), cur_lambdas(i.second[0]), cur_mus(i.second[0]));
						x(i.second[1] * 2 + 1) = convert_to_nu(state.mesh->is_volume(), cur_lambdas(i.second[0]), cur_mus(i.second[0]));
					}
					logger().debug("material E nu: {}", x.transpose());
				};
				material_problem->dparam_to_dx = [body_id_map, dof](ElasticProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
					const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
					const auto &cur_mus = state.assembler.lame_params().mu_mat_;
					dx.setZero(body_id_map.size() * 2);

					for (int e = 0; e < dof; e++)
					{
						const int body_id = state.mesh->get_body_id(e);
						const double E = convert_to_E(state.mesh->is_volume(), cur_lambdas(e), cur_mus(e));
						const double nu = convert_to_nu(state.mesh->is_volume(), cur_lambdas(e), cur_mus(e));
						// const double dlambda_dnu = E * (1 + 2 * nu * nu) / pow(2 * nu * nu + nu - 1, 2);
						// const double dmu_dnu = -E / 2 / pow(1 + nu, 2);
						// const double dlambda_dE = nu / (2 * nu * nu + nu - 1);
						// const double dmu_dE = 1 / 2 / (1 + nu);
						Eigen::Matrix2d jacobian = d_lambda_mu_d_E_nu(state.mesh->is_volume(), E, nu);

						if (!body_id_map.count(body_id))
							continue;
						dx(body_id_map.at(body_id)[1] * 2 + 0) += dparams(e) * jacobian(0, 0) + dparams(e + dof) * jacobian(1, 0);
						dx(body_id_map.at(body_id)[1] * 2 + 1) += dparams(e) * jacobian(0, 1) + dparams(e + dof) * jacobian(1, 1);
					}
				};
			}
			else if (material_params["restriction"].get<std::string>() == "constant_log")
			{
				logger().info("{} objects found, each object has constant material parameters...", body_id_map.size());

				material_problem->x_to_param = [body_id_map, dof](const ElasticProblem::TVector &x, State &state) {
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
				material_problem->param_to_x = [body_id_map](ElasticProblem::TVector &x, State &state) {
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
				material_problem->dparam_to_dx = [body_id_map, dof](ElasticProblem::TVector &dx, const Eigen::VectorXd &dparams, State &state) {
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
		}

		material_problem->param_to_x(x_initial, state);
		material_problem->set_optimization_dim(x_initial.size());

		return material_problem;
	}

	std::shared_ptr<OptimizationProblem> setup_friction_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];
		json material_params;

		state.args["output"]["paraview"]["options"]["material"] = true;

		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "friction")
			{
				material_params = param;
				break;
			}
		}

		std::shared_ptr<FrictionProblem> material_problem = std::make_shared<FrictionProblem>(state, j);
		material_problem->param_to_x(x_initial, state);

		return material_problem;
	}

	std::shared_ptr<OptimizationProblem> setup_damping_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];
		json material_params;

		state.args["output"]["paraview"]["options"]["material"] = true;

		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "damping")
			{
				material_params = param;
				break;
			}
		}

		std::shared_ptr<DampingProblem> material_problem = std::make_shared<DampingProblem>(state, j);
		material_problem->param_to_x(x_initial, state);

		return material_problem;
	}

	std::shared_ptr<OptimizationProblem> setup_shape_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{

		const auto &opt_params = state.args["optimization"];

		std::shared_ptr<ShapeProblem> shape_problem = std::make_shared<ShapeProblem>(state, j);

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state.get_vf(V, F);

		const auto &shape_params = shape_problem->get_shape_params();
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
	
	std::shared_ptr<OptimizationProblem> setup_control_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];
		const auto &opt_nl_params = opt_params["solver"]["nonlinear"];

		std::shared_ptr<ControlProblem> control_problem = std::make_shared<ControlProblem>(state, j);
		std::shared_ptr<cppoptlib::NonlinearSolver<ControlProblem>> nlsolver = make_nl_solver<ControlProblem>(opt_nl_params);

		Eigen::VectorXd x;
		x.setZero(control_problem->get_optimize_boundary_ids_to_position().size() * state.mesh->dimension() * state.args["time"]["time_steps"].get<int>());
		std::vector<json> dirichlet_bc_params = state.args["boundary_conditions"]["dirichlet_boundary"];
		for (int i = 0; i < dirichlet_bc_params.size(); ++i)
			if (control_problem->get_optimize_boundary_ids_to_position().count(dirichlet_bc_params[i]["id"].get<int>()) != 0)
			{
				int position = control_problem->get_optimize_boundary_ids_to_position().at(dirichlet_bc_params[i]["id"].get<int>());
				for (int k = 0; k < state.mesh->dimension(); ++k)
					for (int t = 0; t < state.args["time"]["time_steps"]; ++t)
						x(t * control_problem->get_optimize_boundary_ids_to_position().size() * state.mesh->dimension() + position * state.mesh->dimension() + k) = state.args["boundary_conditions"]["dirichlet_boundary"][i]["value"][k][t].get<double>();
			}
		// logger().info("Starting x: {}", x);

		x_initial = x;

		return control_problem;
	}

	std::shared_ptr<OptimizationProblem> setup_optimization(const std::string &type, State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		std::map<std::string, std::function<std::shared_ptr<OptimizationProblem>(State &, const std::shared_ptr<CompositeFunctional>, Eigen::VectorXd &)>> setup_functions{{"shape", setup_shape_optimization}, {"initial", setup_initial_condition_optimization}, {"control", setup_control_optimization}, {"material", setup_material_optimization}, {"friction", setup_friction_optimization}, {"damping", setup_damping_optimization}};

		return setup_functions[type](state, j, x_initial);
	}

	void single_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		const auto &opt_nl_params = state.args["optimization"]["solver"]["nonlinear"];
		std::shared_ptr<cppoptlib::NonlinearSolver<OptimizationProblem>> nlsolver = make_nl_solver<OptimizationProblem>(opt_nl_params);

		assert(state.args["optimization"]["parameters"].size() == 1);

		Eigen::VectorXd x;
		auto opt_problem = setup_optimization(state.args["optimization"]["parameters"][0]["type"], state, j, x);

		nlsolver->minimize(*opt_problem, x);

		json solver_info;
		nlsolver->get_info(solver_info);
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
			Eigen::VectorXd tmp;
			problems.push_back(setup_optimization(param["type"], state, j, tmp));
			x_initial_size += tmp.size();
			x_initial_list.push_back(tmp);
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
			general_optimization_problem = std::make_shared<GeneralOptimizationProblem>(problems, j);

		return general_optimization_problem;
	}

	void general_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		std::shared_ptr<cppoptlib::NonlinearSolver<GeneralOptimizationProblem>> nlsolver = make_nl_solver<GeneralOptimizationProblem>(state.args["optimization"]["solver"]["nonlinear"]);

		Eigen::VectorXd x;
		auto general_optimization_problem = setup_general_optimization(state, j, x);
		nlsolver->minimize(*general_optimization_problem, x);

		json solver_info;
		nlsolver->get_info(solver_info);
		std::cout << solver_info << std::endl;
	}

	std::shared_ptr<State> create_state(const json &args, const int max_threads)
	{
		std::shared_ptr<State> state = std::make_shared<State>(max_threads);
		state->init_logger("", spdlog::level::level_enum::err, false);
		state->init(args, false);
		state->args["optimization"]["enabled"] = true;
		state->load_mesh();
		Eigen::MatrixXd sol, pressure;
		state->build_basis();
		state->assemble_rhs();
		state->assemble_stiffness_mat();

		return state;
	}

	void solve_pde(State &state)
	{
		state.assemble_rhs();
		state.assemble_stiffness_mat();
		Eigen::MatrixXd sol, pressure;
		state.solve_problem(sol, pressure);
	}

	json apply_opt_json_spec(const json &input_args, bool strict_validation)
	{
		json args_in = input_args;

		// CHECK validity json
		json rules;
		jse::JSE jse;
		{
			jse.strict = strict_validation;
			const std::string polyfem_input_spec = POLYFEM_OPT_INPUT_SPEC;
			std::ifstream file(polyfem_input_spec);

			if (file.is_open())
				file >> rules;
			else
			{
				logger().error("unable to open {} rules", polyfem_input_spec);
				throw std::runtime_error("Invald spec file");
			}
		}

		const bool valid_input = jse.verify_json(args_in, rules);

		if (!valid_input)
		{
			logger().error("invalid input json:\n{}", jse.log2str());
			throw std::runtime_error("Invald input json file");
		}

		json args = jse.inject_defaults(args_in, rules);
		return args;
	}

	std::shared_ptr<AdjointNLProblem> make_nl_problem(json &opt_args)
	{
		opt_args = apply_opt_json_spec(opt_args, false);

		// create states
		json state_args = opt_args["states"];
		assert(state_args.is_array() && state_args.size() > 0);
		std::vector<std::shared_ptr<State>> states(state_args.size());
		int i = 0;
		for (const json &args : state_args)
		{
			json cur_args;
			if (!load_json(args["path"], cur_args))
				log_and_throw_error("Can't find json for State {}", i);

			states[i++] = create_state(cur_args);
		}

		// create parameters
		json param_args = opt_args["parameters"];
		assert(param_args.is_array() && param_args.size() > 0);
		std::vector<std::shared_ptr<Parameter>> parameters(param_args.size());
		i = 0;
		for (const json &args : param_args)
		{
			std::vector<std::shared_ptr<State>> some_states;
			for (int id : args["states"])
			{
				some_states.push_back(states[id]);
			}
			parameters[i++] = Parameter::create(args, some_states);
		}

		// create objectives
		json obj_args = opt_args["functionals"];
		assert(obj_args.is_array() && obj_args.size() > 0);
		std::vector<std::shared_ptr<Objective>> objs(obj_args.size());
		Eigen::VectorXd weights;
		weights.setOnes(objs.size());
		i = 0;
		for (const json &args : obj_args)
		{
			weights[i] = args["weight"];
			objs[i++] = Objective::create(args, parameters, states);
		}
		std::shared_ptr<SumObjective> sum_obj = std::make_shared<SumObjective>(objs, weights);

		std::shared_ptr<AdjointNLProblem> nl_problem = std::make_shared<AdjointNLProblem>(sum_obj, parameters, states, opt_args);

		return nl_problem;
	}
} // namespace polyfem