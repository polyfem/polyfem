#include <CLI/CLI.hpp>
#include <polyfem/Optimizations.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polyfem/StringUtils.hpp>
#include <polyfem/Logger.hpp>

#include <polyfem/ImplicitTimeIntegrator.hpp>
#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/JSONUtils.hpp>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <filesystem>
#include <time.h>

using namespace polyfem;
using namespace polysolve;
using namespace Eigen;

bool has_arg(const CLI::App &command_line, const std::string &value)
{
	const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

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
}

void print_centers(const Eigen::MatrixXd &centers, const std::vector<bool> &active_mask)
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
}

void vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat)
{
	int size = sqrt(vec.size());
	assert(size * size == vec.size());

	mat.resize(size, size);
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			mat(i, j) = vec(i * size + j);
}

int main(int argc, char **argv)
{
	using namespace std::filesystem;

	CLI::App command_line{"polyfem"};
	// Eigen::setNbThreads(1);

	// Input
	std::string mesh_file = "";
	std::string json_file = "";
	std::string febio_file = "";

	// Output
	std::string output_dir = "";
	std::string output_json = "";
	std::string output_vtu = "";
	std::string screenshot = "";

	// Problem
	std::string problem_name = "";
	std::string scalar_formulation = "";
	std::string tensor_formulation = "";
	std::string time_integrator_name = "";
	std::string solver = "";

	std::string bc_method = "";

	int n_refs = 0;
	bool count_flipped_els = false;

	std::string log_file = "";
	bool is_quiet = false;
	bool stop_after_build_basis = false;
	bool lump_mass_mat = false;
	spdlog::level::level_enum log_level = spdlog::level::debug;
	int nl_solver_rhs_steps = 1;
	int cache_size = -1;
	size_t max_threads = std::numeric_limits<size_t>::max();
	double f_delta = 0;

	bool use_al = false;
	int min_component = -1;

	double vis_mesh_res = -1;

	std::string target_path = "";
	std::string opt_type = "";
	std::string target_type = "";

	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);
	command_line.add_option("-m,--mesh", mesh_file, "Mesh path")->check(CLI::ExistingFile);
	command_line.add_option("-b,--febio", febio_file, "FEBio file path")->check(CLI::ExistingFile);
	command_line.add_option("--target_path", target_path, "target path")->check(CLI::ExistingFile);

	const std::vector<std::string> solvers = LinearSolver::availableSolvers();
	command_line.add_option("--solver", solver, "Linear solver to use")->check(CLI::IsMember(solvers));

	command_line.add_flag("--al", use_al, "Use augmented lagrangian");
	command_line.add_flag("--count_flipped_els", count_flipped_els, "Count flippsed elements");

	const std::vector<std::string> bc_methods = {"", "sample", "lsq"}; //, "integrate"};
	command_line.add_option("--bc_method", bc_method, "Method used for boundary conditions")->check(CLI::IsMember(bc_methods));

	const std::set<std::string> opt_types = {"material", "shape", "initial"};
	// const std::vector<std::string> opt_types = {"material", "shape", "initial"};
	// command_line.add_option("--type", opt_type, "opt type")->check(CLI::IsMember(opt_types));

	const std::set<std::string> target_types = {"exact", "exact-center", "sine", "max-height", "center-data", "last-center", "marker-data"};
	// const std::vector<std::string> target_types = {"exact", "exact-center", "sine", "max-height", "center-data", "last-center", "marker-data"};
	// command_line.add_option("--target", target_type, "functional type")->check(CLI::IsMember(target_types));

	// IO
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	command_line.add_flag("--quiet", is_quiet, "Disable cout for logging");
	command_line.add_option("--log_file", log_file, "Log to a file");
	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off}};
	command_line.add_option("--log_level", log_level, "Log level")
		->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

	CLI11_PARSE(command_line, argc, argv);

	json in_args = json({});
	json target_in_args = json({});

	Eigen::Vector3d target_position;
	target_position.setZero();
	if (!json_file.empty())
	{
		std::ifstream file(json_file);

		if (file.is_open())
			file >> in_args;
		else
			logger().error("unable to open {} file", json_file);
		file.close();

		in_args["root_path"] = json_file;

		if (in_args.contains("default_params"))
		{
			apply_default_params(in_args);
			in_args.erase("default_params"); // Remove this so state does not redo the apply
		}

		if (in_args["optimization"].contains("parameter"))
			opt_type = in_args["optimization"]["parameter"];
		assert(opt_types.count(opt_type));

		if (in_args["optimization"].contains("trajectory"))
		{
			auto trajectory_params = in_args["optimization"]["trajectory"];

			if (trajectory_params.contains("type"))
				target_type = trajectory_params["type"];
			assert(target_types.count(target_type));

			if (trajectory_params.contains("path"))
				target_path = trajectory_params["path"];

			if (target_type == "last-center")
			{
				assert(trajectory_params.contains("target_position"));
				int i = 0;
				for (double x : trajectory_params["target_position"].get<std::vector<double>>())
				{
					if (i >= target_position.size())
						break;
					target_position(i) = x;
					i++;
				}
			}
		}
	}
	else
	{
		logger().error("No json provided!");
		return EXIT_FAILURE;
	}

	if (target_type == "exact" || target_type == "exact-center")
	{
		if (!target_path.empty())
		{
			std::ifstream file(target_path);

			if (file.is_open())
				file >> target_in_args;
			else
				logger().error("unable to open {} file", target_path);
			file.close();

			target_in_args["root_path"] = target_path;

			if (target_in_args.contains("default_params"))
			{
				apply_default_params(target_in_args);
				target_in_args.erase("default_params"); // Remove this so state does not redo the apply
			}
		}
		else
			logger().error("Target json input missing!");
	}

	if (!solver.empty())
		in_args["solver_type"] = solver;

	if (!output_dir.empty())
		create_directories(output_dir);

	if (opt_type == "material")
		in_args["export"]["material_params"] = true;

	// compute reference solution
	State state_reference(max_threads);
	if (target_type == "exact" || target_type == "exact-center")
	{
		logger().info("Start reference solve...");
		state_reference.init_logger(log_file, log_level, is_quiet);
		state_reference.init(target_in_args, output_dir);
		state_reference.load_mesh();

		// Mesh was not loaded successfully; load_mesh() logged the error.
		if (state_reference.mesh == nullptr)
		{
			// Cannot proceed without a mesh.
			return EXIT_FAILURE;
		}

		state_reference.compute_mesh_stats();
		state_reference.build_basis();

		const int cur_log = state_reference.current_log_level;
		state_reference.set_log_level(in_args["optimization"].contains("solve_log_level") ? in_args["optimization"]["solve_log_level"].get<int>() : cur_log);
		state_reference.args["save_time_sequence"] = false;
		state_reference.assemble_rhs();
		state_reference.assemble_stiffness_mat();
		state_reference.solve_problem();
		state_reference.set_log_level(cur_log);

		if (!state_reference.problem->is_time_dependent())
			state_reference.save_vtu(state_reference.resolve_output_path("target.vtu"), 0.);

		logger().info("Reference solve done!");
	}

	State state(max_threads);
	state.init_logger(log_file, log_level, is_quiet);
	state.init(in_args, output_dir);
	state.load_mesh();

	if (state.args["has_collision"] && !state.args.contains("barrier_stiffness"))
	{
		logger().error("Not fixing the barrier stiffness!");
		return EXIT_FAILURE;
	}

	// Mesh was not loaded successfully; load_mesh() logged the error.
	if (state.mesh == nullptr)
	{
		// Cannot proceed without a mesh.
		return EXIT_FAILURE;
	}

	std::set<int> interested_ids;
	if (in_args.contains("meshes") && !in_args["meshes"].empty())
	{
		const auto &meshes = in_args["meshes"].get<std::vector<json>>();
		for (const auto &m : meshes)
		{
			if (m.contains("interested") && m["interested"].get<bool>())
			{
				if (!m.contains("body_id"))
				{
					logger().error("No body id in interested mesh!");
				}
				interested_ids.insert(m["body_id"].get<int>());
			}
		}
	}

	state.compute_mesh_stats();
	state.build_basis();

	const int cur_log = state.current_log_level;
	state.set_log_level(in_args["optimization"].contains("solve_log_level") ? in_args["optimization"]["solve_log_level"].get<int>() : cur_log);
	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.set_log_level(cur_log);

	assert(state.formulation() == "LinearElasticity" || state.formulation() == "NeoHookean");
	std::shared_ptr<CompositeFunctional> func;
	if (target_type == "exact")
		func = CompositeFunctional::create("Trajectory");
	else if (target_type == "exact-center")
		func = CompositeFunctional::create("CenterTrajectory");
	else if (target_type == "last-center")
		func = CompositeFunctional::create("CenterTrajectory");
	else if (target_type == "sine")
		func = CompositeFunctional::create("TargetY");
	else if (target_type == "max-height")
		func = CompositeFunctional::create("Height");
	else if (target_type == "center-data")
		func = CompositeFunctional::create("CenterXYTrajectory");
	else if (target_type == "marker-data")
		func = CompositeFunctional::create("NodeTrajectory");
	else
		logger().error("Invalid target type!");

	func->set_interested_ids(interested_ids);

	if (target_type == "exact")
	{
		auto &f = *dynamic_cast<TrajectoryFunctional *>(func.get());
		f.set_reference(&state_reference, state);
	}
	else if (target_type == "exact-center")
	{
		auto &f = *dynamic_cast<CenterTrajectoryFunctional *>(func.get());
		std::vector<Eigen::VectorXd> barycenters;
		f.get_barycenter_series(state_reference, barycenters);
		f.set_center_series(barycenters);
		std::cout << "Centers: ";
		for (auto x : barycenters)
			std::cout << x.transpose() << ", ";
		std::cout << "\n";
	}
	else if (target_type == "last-center")
	{
		auto &f = *dynamic_cast<CenterTrajectoryFunctional *>(func.get());
		f.set_transient_integral_type("final");
		std::vector<Eigen::VectorXd> barycenters(1);
		barycenters[0] = target_position;
		f.set_center_series(barycenters);
	}
	else if (target_type == "sine")
	{
		auto &f = *dynamic_cast<TargetYFunctional *>(func.get());
		f.set_target_function([](const double x) {
			return sin(x) * 0.7;
		});
		f.set_target_function_derivative([](const double x) {
			return cos(x) * 0.7;
		});
	}
	else if (target_type == "center-data")
	{
		auto &f = *dynamic_cast<CenterXYTrajectoryFunctional *>(func.get());
		std::ifstream infile(target_path);
		std::vector<Eigen::VectorXd> centers;
		double x = 0, y = 0;
		Eigen::VectorXd center;
		center.setZero(3);
		const int down_sample_rate = 1; // pick one line in every 1 lines
		int n = -1;
		while (infile.good())
		{
			infile >> x;
			infile >> y;
			n++;
			if (n % down_sample_rate != 0)
				continue;
			center(0) = x / 100;
			center(1) = y / 100; // cm to m
			centers.push_back(center);
		}
		infile.close();
		f.set_center_series(centers);
		print_centers(centers);
	}
	else if (target_type == "marker-data")
	{
		const std::string scene = state.args["optimization"]["name"];
		auto &f = *dynamic_cast<NodeTrajectoryFunctional *>(func.get());
		std::ifstream infile(target_path);
		std::vector<Eigen::VectorXd> markers;
		std::vector<Eigen::VectorXd> marker_rest_position;
		Eigen::VectorXd center(3);
		center << 0, 0, 0;
		while (infile.good())
		{
			if (scene == "Unit-Cell")
			{
				infile >> center(0);
				infile >> center(1);
				infile >> center(2);
				marker_rest_position.push_back(center);
			}

			infile >> center(0);
			infile >> center(1);
			infile >> center(2);
			markers.push_back(center);
		}
		infile.close();

		if (scene != "Unit-Cell")
		{
			if (markers.size() != 25)
				logger().error("Wrong sample number for compressed cube!");
			marker_rest_position.resize(25);
			for (int y = 0; y < 5; y++)
				for (int z = 0; z < 5; z++)
				{
					center << 0.02, -0.02 + 0.01 * y, -0.02 + 0.01 * z;
					marker_rest_position[y + 5 * z] = center;
				}
		}

		// markers to nodes
		Eigen::MatrixXd targets;
		targets.setZero(state.n_bases, state.mesh->dimension());
		std::vector<bool> active_mask(state.n_bases, false);
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state.get_vf(V, F, false);
		assert(targets.rows() == V.rows());
		for (int s = 0; s < marker_rest_position.size(); s++)
		{
			if (scene != "Unit-Cell")
				if (s == 3) // wrong data
					continue;
			double min_dist = std::numeric_limits<double>::max();
			int min_dist_id = -1;
			for (int v = 0; v < V.rows(); v++)
			{
				if ((V.row(v) - marker_rest_position[s].transpose()).norm() < min_dist)
				{
					min_dist_id = v;
					min_dist = (V.row(v) - marker_rest_position[s].transpose()).norm();
				}
			}
			if (active_mask[min_dist_id])
				logger().error("Same node has different markers!!");
			else if (min_dist > 1e-4)
			{
				logger().error("Too large err {} between {} and {}!!", min_dist, V.row(min_dist_id), marker_rest_position[s].transpose());
			}
			targets.row(min_dist_id) = markers[s];
			active_mask[min_dist_id] = true;
		}
		f.set_active_vertex_mask(active_mask);
		f.set_target_vertex_positions(targets);
		print_centers(targets, active_mask);
	}

	// shape optimization
	if (opt_type == "shape")
		shape_optimization(state, func, state.args["optimization"]);
	else if (opt_type == "material")
		material_optimization(state, func, state.args["optimization"]);
	else if (opt_type == "initial")
		initial_condition_optimization(state, func, state.args["optimization"]);
	else
		logger().error("Invalid optimization type!");

	return EXIT_SUCCESS;
}
