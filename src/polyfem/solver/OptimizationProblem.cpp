#include <polyfem/OptimizationProblem.hpp>

namespace polyfem
{
	namespace
	{
		void print_centers(const std::vector<Eigen::VectorXd> &centers)
		{
			std::cout << "[";
			for (int c = 0; c < centers.size(); c++)
			{
				const auto &center = centers[c];
				std::cout << "[";
				for (int d = 0; d < center.size(); d++)
				{
					std::cout << center(d);
					if (d < center.size() - 1)
						std::cout << ", ";
				}
				if (c < centers.size() - 1)
					std::cout << "],";
				else
					std::cout << "]";
			}
			std::cout << "]\n";
		};

		void print_markers(const Eigen::MatrixXd &centers, const std::vector<bool> &active_mask)
		{
			std::cout << "[";
			for (int c = 0; c < centers.rows(); c++)
			{
				if (!active_mask[c])
					continue;
				std::cout << "[";
				for (int d = 0; d < centers.cols(); d++)
				{
					std::cout << centers(c, d);
					if (d < centers.cols() - 1)
						std::cout << ", ";
				}
				if (c < centers.rows() - 1)
					std::cout << "],";
				else
					std::cout << "]";
			}
			std::cout << "]\n";
		};
	} // namespace

	OptimizationProblem::OptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_, const json &args) : state(state_)
	{
		opt_params = args;
		j = j_;
		dim = state.mesh->dimension();
		actual_dim = state.problem->is_scalar() ? 1 : dim;
		opt_params = args;

		save_freq = args.contains("save_frequency") ? args["save_frequency"].get<int>() : 1;

		cur_grad.resize(0);
		cur_val = std::nan("");
	}

	void OptimizationProblem::solve_pde(const TVector &x)
	{
		if (!state.problem->is_time_dependent())
			if (optimization_name == "shape")
			{
				if (x_at_ls_begin.size() == x.size())
				{
					logger().debug("Use better initial guess...");
					if (sol_at_ls_begin.size() == x.size())
						state.pre_sol = sol_at_ls_begin + x_at_ls_begin - x;
					else if (sol_at_ls_begin.size() == state.n_bases)
						state.pre_sol = sol_at_ls_begin + state.down_sampling_mat.transpose() * (x_at_ls_begin - x);
				}
			}
			else if (sol_at_ls_begin.size() > 0)
			{
				logger().debug("Use better initial guess...");
				state.pre_sol = sol_at_ls_begin;
			}

		// control forward solve log level
		const int cur_log = state.current_log_level;
		state.set_log_level(opt_params.contains("solve_log_level") ? opt_params["solve_log_level"].get<int>() : cur_log);

		auto output_dir = state.output_dir;
		if (state.problem->is_time_dependent() && save_iter < iter)
		{
			save_iter++;
			state.output_dir = "iter_" + std::to_string(iter);
			if (std::filesystem::exists(state.output_dir))
				std::filesystem::remove_all(state.output_dir);
			std::filesystem::create_directories(state.output_dir);
			logger().info("Save time sequence to {} ...", state.output_dir);
		}

		state.assemble_rhs();
		state.assemble_stiffness_mat();
		state.solve_problem();

		if (optimization_name != "shape")
			sol_at_ls_begin = state.sol;

		if (j->get_functional_name() == "CenterTrajectory")
		{
			CenterTrajectoryFunctional f;
			f.set_interested_ids(j->get_interested_ids());
			std::vector<Eigen::VectorXd> barycenters;
			f.get_barycenter_series(state, barycenters);
			print_centers(barycenters);
		}
		else if (j->get_functional_name() == "NodeTrajectory")
		{
			const auto &f = *dynamic_cast<NodeTrajectoryFunctional *>(j.get());
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			state.get_vf(V, F, false);
			V += unflatten(state.sol, state.mesh->dimension());
			print_markers(V, f.get_active_vertex_mask());
		}

		state.output_dir = output_dir;
		state.set_log_level(cur_log);
	}

	void OptimizationProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		descent_direction = x1 - x0;

		// debug
		if (opt_params.contains("debug_fd") && opt_params["debug_fd"].get<bool>())
		{
			double t = 1e-6;
			TVector new_x = x0 + descent_direction * t;

			solution_changed(new_x);
			double J2 = value(new_x);

			solution_changed(x0);
			double J1 = value(x0);
			TVector gradv;
			gradient(x0, gradv);

			logger().debug("step size: {}, finite difference: {}, derivative: {}", t, (J2 - J1) / t, gradv.dot(descent_direction));
		}
	}
} // namespace polyfem