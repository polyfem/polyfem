#include "AdjointNLProblem.hpp"

#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/optimization/VarFormDiff.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/forms/AdjointForm.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/MshWriter.hpp>
#include <polyfem/mesh/SlimSmooth.hpp>

#include <Eigen/Core>
#include <spdlog/fmt/fmt.h>

#include <list>
#include <stack>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

namespace polyfem::solver
{
	namespace
	{

		Eigen::VectorXd get_updated_mesh_nodes(const VariableToSimulationGroup &variables_to_simulation, const std::shared_ptr<varform::DifferentiableVarForm> &current_varform, const Eigen::VectorXd &x)
		{
			Eigen::MatrixXd V;
			current_varform->get_vertices(V);
			Eigen::VectorXd X = utils::flatten(V);

			variables_to_simulation.compute_state_variable(ParameterType::Shape, *current_varform, x, X);
			variables_to_simulation.compute_state_variable(ParameterType::PeriodicShape, *current_varform, x, X);

			return X;
		}

		// Class to represent a graph
		class Graph
		{
			int V; // No. of vertices'

			// adjacency lists
			std::vector<std::list<int>> adj;

			// A function used by topologicalSort
			void topologicalSortUtil(int v, std::vector<bool> &visited, std::stack<int> &Stack);

		public:
			Graph(int V); // Constructor

			// function to add an edge to graph
			void addEdge(int v, int w);

			// prints a Topological Sort of the complete graph
			std::vector<int> topologicalSort();
		};

		Graph::Graph(int V)
		{
			this->V = V;
			adj.resize(V);
		}

		void Graph::addEdge(int v, int w)
		{
			adj[v].push_back(w); // Add w to v’s list.
		}

		// A recursive function used by topologicalSort
		void Graph::topologicalSortUtil(int v, std::vector<bool> &visited,
										std::stack<int> &Stack)
		{
			// Mark the current node as visited.
			visited[v] = true;

			// Recur for all the vertices adjacent to this vertex
			std::list<int>::iterator i;
			for (i = adj[v].begin(); i != adj[v].end(); ++i)
				if (!visited[*i])
					topologicalSortUtil(*i, visited, Stack);

			// Push current vertex to stack which stores result
			Stack.push(v);
		}

		// The function to do Topological Sort. It uses recursive
		// topologicalSortUtil()
		std::vector<int> Graph::topologicalSort()
		{
			std::stack<int> Stack;

			// Mark all the vertices as not visited
			std::vector<bool> visited(V, false);

			// Call the recursive helper function to store Topological
			// Sort starting from all vertices one by one
			for (int i = 0; i < V; i++)
				if (visited[i] == false)
					topologicalSortUtil(i, visited, Stack);

			// Print contents of stack
			std::vector<int> sorted;
			while (Stack.empty() == false)
			{
				sorted.push_back(Stack.top());
				Stack.pop();
			}

			return sorted;
		}
	} // namespace

	AdjointNLProblem::AdjointNLProblem(std::shared_ptr<AdjointForm> form,
									   const VariableToSimulationGroup &variables_to_simulation,
									   const std::vector<std::shared_ptr<varform::DifferentiableVarForm>> &all_varforms,
									   const std::vector<std::shared_ptr<DiffCache>> &all_diff_caches,
									   const json &args,
									   std::function<bool()> remeshing_trigger)
		: FullNLProblem({form}),
		  form_(form),
		  variables_to_simulation_(variables_to_simulation),
		  all_varforms_(all_varforms),
		  all_diff_caches_(all_diff_caches),
		  save_freq(args["output"]["save_frequency"]),
		  enable_slim(args["solver"]["advanced"]["enable_slim"]),
		  smooth_line_search(args["solver"]["advanced"]["smooth_line_search"]),
		  solve_in_parallel(args["solver"]["advanced"]["solve_in_parallel"]),
		  remeshing_trigger_(std::move(remeshing_trigger))
	{
		cur_grad.setZero(0);

		if (enable_slim && args["solver"]["nonlinear"]["advanced"]["apply_gradient_fd"] != "None")
			adjoint_logger().warn("SLIM may affect the finite difference result!");

		if (enable_slim && smooth_line_search)
			adjoint_logger().warn("Both in-line-search SLIM and after-line-search SLIM are ON!");

		if (args["output"]["solution"] != "")
		{
			solution_ostream.open(args["output"]["solution"].get<std::string>(), std::ofstream::out);
			if (!solution_ostream.is_open())
				adjoint_logger().error("Cannot open solution file for writing!");
		}

		solve_in_order.clear();
		{
			Graph G(all_varforms.size());
			for (int k = 0; k < all_varforms.size(); k++)
			{
				auto &arg = args["states"][k];
				if (arg["initial_guess"].get<int>() >= 0)
					G.addEdge(arg["initial_guess"].get<int>(), k);
			}

			solve_in_order = G.topologicalSort();
		}

		active_varform_mask.assign(all_varforms_.size(), false);
		for (int i = 0; i < all_varforms_.size(); i++)
		{
			for (const auto &v2sim : variables_to_simulation_.data)
			{
				if (v2sim->affects_varform(*all_varforms_[i]))
				{
					active_varform_mask[i] = true;
					break;
				}
			}
		}
	}

	AdjointNLProblem::AdjointNLProblem(
		std::shared_ptr<AdjointForm> form,
		const std::vector<std::shared_ptr<AdjointForm>> &stopping_conditions,
		const VariableToSimulationGroup &variables_to_simulation,
		const std::vector<std::shared_ptr<varform::DifferentiableVarForm>> &all_varforms,
		const std::vector<std::shared_ptr<DiffCache>> &all_diff_caches,
		const json &args,
		std::function<bool()> remeshing_trigger)
		: AdjointNLProblem(
			  form, variables_to_simulation, all_varforms, all_diff_caches, args,
			  std::move(remeshing_trigger))
	{
		stopping_conditions_ = stopping_conditions;
	}

	void AdjointNLProblem::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		log_and_throw_adjoint_error("Hessian not supported!");
	}

	double AdjointNLProblem::value(const Eigen::VectorXd &x)
	{
		return form_->value(x);
	}

	void AdjointNLProblem::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
	{
		if (cur_grad.size() == x.size())
			gradv = cur_grad;
		else
		{
			gradv.setZero(x.size());

			{
				POLYFEM_SCOPED_TIMER("adjoint solve");
				for (int i = 0; i < all_varforms_.size(); i++)
					solve_adjoint_cached(*all_varforms_[i], *all_diff_caches_[i], form_->compute_reduced_adjoint_rhs(x, *all_varforms_[i], *all_diff_caches_[i]));
			}

			{
				POLYFEM_SCOPED_TIMER("gradient assembly");
				form_->first_derivative(x, gradv);
				if (x.size() < 10)
				{
					adjoint_logger().trace("x {}", x.transpose());
					adjoint_logger().trace("gradient {}", gradv.transpose());
				}
			}

			cur_grad = gradv;
		}
	}

	bool AdjointNLProblem::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		bool need_rebuild_basis = false;

		// update to new parameter and check if the new parameter is valid to solve
		for (const auto &v : variables_to_simulation_.data)
			if (v->parameter_type() == ParameterType::Shape || v->parameter_type() == ParameterType::PeriodicShape)
				need_rebuild_basis = true;

		if (need_rebuild_basis && smooth_line_search)
		{
			Eigen::MatrixXd X, V0, V1;
			Eigen::MatrixXi F;

			for (auto varform : all_varforms_)
			{

				V1 = utils::unflatten(
					get_updated_mesh_nodes(variables_to_simulation_, varform, x1),
					varform->get_mesh().dimension());
				varform->get_vertices(V0);
				varform->get_elements(F);

				Eigen::MatrixXd V_smooth;
				bool slim_success = polyfem::mesh::apply_slim(V0, F, V1, V_smooth);
				if (!slim_success)
				{
					adjoint_logger().info("SLIM failed, step not valid!");
					return false;
				}

				V1 = V_smooth;

				bool flipped = utils::is_flipped(V1, F);
				if (flipped)
				{
					adjoint_logger().info("Found flipped element in LS, step not valid!");
					return false;
				}
			}
		}

		return form_->is_step_valid(x0, x1);
	}

	bool AdjointNLProblem::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		return form_->is_step_collision_free(x0, x1);
	}

	double AdjointNLProblem::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		return form_->max_step_size(x0, x1);
	}

	void AdjointNLProblem::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		form_->line_search_begin(x0, x1);
	}

	void AdjointNLProblem::line_search_end()
	{
		form_->line_search_end();
	}

	void AdjointNLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		save_to_file(save_iter++, data.x);

		form_->post_step(data);
	}

	void AdjointNLProblem::save_to_file(const int iter_num, const Eigen::VectorXd &x0)
	{
		int id = 0;

		if (solution_ostream.is_open())
		{
			adjoint_logger().debug("Save solution at iteration {} to file...", iter_num);
			solution_ostream << iter_num << ": " << std::setprecision(16) << x0.transpose() << std::endl;
			solution_ostream.flush();
		}

		if (iter_num % save_freq != 0)
			return;
		adjoint_logger().info("Saving iteration {}", iter_num);
		for (int i = 0; i < all_varforms_.size(); ++i)
		{
			auto &varform = all_varforms_[i];
			auto &diff_cache = all_diff_caches_[i];

			bool save_vtu = true;
			bool save_rest_mesh = true;

			std::string vis_mesh_path = varform->output_file_path(fmt::format("opt_state_{:d}_iter_{:d}.vtu", id, iter_num));
			std::string mesh_ext = varform->get_mesh().is_volume() ? ".msh" : ".obj";
			std::string rest_mesh_path = varform->output_file_path(fmt::format("opt_state_{:d}_iter_{:d}" + mesh_ext, id, iter_num));
			id++;

			if (!save_vtu)
				continue;
			adjoint_logger().debug("Save final vtu to file {} ...", vis_mesh_path);

			double tend = varform->get_args().value("tend", 1.0);
			double dt = 1;
			if (!varform->get_args()["time"].is_null())
				dt = varform->get_args()["time"]["dt"];

			Eigen::MatrixXd sol = diff_cache->u(-1);

			varform->save_vtu(vis_mesh_path, sol, tend, dt);

			if (!save_rest_mesh)
				continue;
			adjoint_logger().debug("Save rest mesh to file {} ...", rest_mesh_path);

			// If shape opt, save rest meshes as well
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			varform->get_vertices(V);
			varform->get_elements(F);
			if (varform->get_mesh().is_volume())
				io::MshWriter::write(rest_mesh_path, V, F, varform->get_mesh().get_body_ids(), true, false);
			else
				io::OBJWriter::write(rest_mesh_path, V, F);
		}
	}

	void AdjointNLProblem::solution_changed(const Eigen::VectorXd &newX)
	{
		bool need_rebuild_basis = false;

		// update to new parameter and check if the new parameter is valid to solve
		for (const auto &v : variables_to_simulation_.data)
		{
			v->update(newX);
			if (v->parameter_type() == ParameterType::Shape || v->parameter_type() == ParameterType::PeriodicShape)
				need_rebuild_basis = true;
		}

		if (need_rebuild_basis)
		{
			for (const auto &varform : all_varforms_)
				varform->prepare();
		}

		// solve PDE
		solve_pde();

		form_->solution_changed(newX);

		curr_x = newX;
	}

	bool AdjointNLProblem::after_line_search_custom_operation(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		if (!enable_slim)
			return false;

		// SLIM smoothing for shape optimization.

		std::vector<Eigen::MatrixXd> V_old;
		std::vector<Eigen::MatrixXd> V_new;
		for (const auto &varform : all_varforms_)
		{
			V_old.push_back(utils::unflatten(
				get_updated_mesh_nodes(variables_to_simulation_, varform, x0),
				varform->get_mesh().dimension()));
			V_new.push_back(utils::unflatten(
				get_updated_mesh_nodes(variables_to_simulation_, varform, x1),
				varform->get_mesh().dimension()));
		}

		std::vector<Eigen::MatrixXd> V_smooth;
		V_smooth.reserve(all_varforms_.size());
		for (int i = 0; i < all_varforms_.size(); ++i)
		{
			const auto &varform = all_varforms_[i];
			Eigen::MatrixXd V_out;
			Eigen::MatrixXi F;
			varform->get_elements(F);

			if (!polyfem::mesh::apply_slim(V_old[i], F, V_new[i], V_out, 50))
			{
				adjoint_logger().warn("SLIM failed; keeping the accepted unsmoothed step.");
				return false;
			}
			V_smooth.push_back(std::move(V_out));
		}

		for (int i = 0; i < all_varforms_.size(); ++i)
			all_varforms_[i]->set_vertex_positions(V_smooth[i]);

		adjoint_logger().debug("SLIM succeeded!");

		return true;
	}

	void AdjointNLProblem::solve_pde()
	{
		if (solve_in_parallel)
		{
			adjoint_logger().info("Run simulations in parallel...");

			utils::maybe_parallel_for(all_varforms_.size(), [&](int start, int end, int thread_id) {
				for (int i = start; i < end; i++)
				{
					auto &varform = all_varforms_[i];
					auto &diff_cache = all_diff_caches_[i];
					if (active_varform_mask[i] || diff_cache->size() == 0)
					{
						const auto *initial_conditions = diff_cache->initial_condition_override ? &*diff_cache->initial_condition_override : nullptr;
						const varform::ForwardStepCallback post_step = [varform, diff_cache](const int step, const Eigen::MatrixXd &solution) {
							diff_cache->cache_transient(step, *varform, solution, nullptr);
						};
						Eigen::MatrixXd solution;
						varform->solve(solution, initial_conditions, post_step, true);
					}
				}
			});
		}
		else
		{
			adjoint_logger().info("Run simulations in serial...");

			for (int i : solve_in_order)
			{
				auto &varform = all_varforms_[i];
				auto &diff_cache = all_diff_caches_[i];
				if (active_varform_mask[i] || diff_cache->size() == 0)
				{
					const auto *initial_conditions = diff_cache->initial_condition_override ? &*diff_cache->initial_condition_override : nullptr;
					const varform::ForwardStepCallback post_step = [varform, diff_cache](const int step, const Eigen::MatrixXd &solution) {
						diff_cache->cache_transient(step, *varform, solution, nullptr);
					};
					Eigen::MatrixXd solution;
					varform->solve(solution, initial_conditions, post_step, true);
				}
			}
		}

		cur_grad.resize(0);
	}

	bool AdjointNLProblem::stop(const TVector &x)
	{
		if (remeshing_trigger_ && remeshing_trigger_())
			return true;

		if (stopping_conditions_.size() == 0)
			return false;

		for (auto &obj : stopping_conditions_)
		{
			obj->solution_changed(x);
			if (obj->value(x) > 0)
				return false;
		}
		return true;
	}

} // namespace polyfem::solver
