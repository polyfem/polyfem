#include <filesystem>

#include <CLI/CLI.hpp>
#include <polyfem/solver/Optimizations.hpp>

#include <polyfem/State.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polysolve/LinearSolver.hpp>

#include <algorithm>

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
	size_t max_threads = std::min((size_t)32, std::numeric_limits<size_t>::max());
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	std::string json_file = "";
	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);

	std::string hdf5_file = "";
	command_line.add_option("--hdf5", hdf5_file, "Simulation hdf5 file")->check(CLI::ExistingFile);

	std::string output_dir = "";
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	bool is_quiet = false;
	command_line.add_flag("--quiet", is_quiet, "Disable cout for logging");

	std::string log_file = "";
	command_line.add_option("--log_file", log_file, "Log to a file");

	const std::set<std::string> valid_opt_types = {"material", "shape", "initial", "control"};

	const std::set<std::string> matching_types = {"exact-center", "sine", "exact", "sdf", "center-data", "last-center", "marker-data"};

	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off}};
	spdlog::level::level_enum log_level = spdlog::level::debug;
	command_line.add_option("--log_level", log_level, "Log level")
		->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

	CLI11_PARSE(command_line, argc, argv);

	std::vector<std::string> names;
	std::vector<Eigen::MatrixXi> cells;
	std::vector<Eigen::MatrixXd> vertices;

	State state(max_threads);
	state.init_logger(log_file, log_level, false);

	std::string target_path, matching_type, opt_type;
	std::vector<std::string> opt_types;

	json opt_params, objective_params;
	std::string transient_integral_type = "";

	if (!output_dir.empty())
		create_directories(output_dir);

	Eigen::MatrixXd control_points, tangents, delta;
	Eigen::Vector3d target_position;
	target_position.setZero();
	if (!json_file.empty())
	{
		std::ifstream file(json_file);

		json in_args = json({});
		if (file.is_open())
			file >> in_args;
		else
			logger().error("unable to open {} file", json_file);
		file.close();

		in_args["root_path"] = json_file;
		state.init(in_args, false, output_dir);
	}
	else
	{
		logger().error("No json provided!");
		return EXIT_FAILURE;
	}

	opt_params = state.args["optimization"];
	if (opt_params["parameters"].size() > 0)
	{
		if (opt_params["parameters"].size() == 1)
		{
			opt_type = opt_params["parameters"][0]["type"];
			assert(valid_opt_types.count(opt_type));
		}
		else
			for (int i = 0; i < opt_params["parameters"].size(); ++i)
			{
				opt_types.push_back(opt_params["parameters"][i]["type"]);
				assert(valid_opt_types.count(opt_types[i]));
			}
	}
	else
		throw std::runtime_error("No optimization parameter specified!");

	if (opt_params["functionals"].size() > 0)
	{
		objective_params = opt_params["functionals"][0];

		if (objective_params["type"] != "trajectory" && objective_params["type"] != "height")
			throw std::runtime_error("Unrecognized functional!");

		transient_integral_type = objective_params["transient_integral_type"];

		if (objective_params["type"] == "trajectory")
		{
			matching_type = objective_params["matching"];
			target_path = objective_params["path"];

			if (matching_type == "last-center")
			{
				int i = 0;
				for (double x : objective_params["target_position"].get<std::vector<double>>())
				{
					if (i >= target_position.size())
						break;
					target_position(i) = x;
					i++;
				}
			}
			else if (matching_type == "sdf")
			{
				double dim;
				control_points.setZero(objective_params["control_points"].size(), objective_params["control_points"][0].size());
				for (int i = 0; i < objective_params["control_points"].size(); ++i)
				{
					dim = objective_params["control_points"][i].size();
					for (int j = 0; j < objective_params["control_points"][i].size(); ++j)
						control_points(i, j) = objective_params["control_points"][i][j].get<double>();
				}
				tangents.setZero(objective_params["tangents"].size(), objective_params["tangents"][0].size());
				for (int i = 0; i < objective_params["tangents"].size(); ++i)
					for (int j = 0; j < objective_params["tangents"][i].size(); ++j)
						tangents(i, j) = objective_params["tangents"][i][j].get<double>();

				delta.setZero(objective_params["delta"].size(), 1);
				for (int i = 0; i < delta.size(); ++i)
					delta(i) = objective_params["delta"][i].get<double>();
			}
		}
	}
	else
		throw std::runtime_error("No functional specifed in json!");

	State state_reference(max_threads);
	state_reference.init_logger(log_file, state.args["optimization"]["output"]["solve_log_level"], false);
	if (objective_params["type"] == "trajectory" && utils::StringUtils::startswith(matching_type, "exact"))
	{
		if (!target_path.empty())
		{
			std::ifstream file(target_path);

			json target_in_args = json({});
			if (file.is_open())
				file >> target_in_args;
			else
				logger().error("unable to open {} file", target_path);
			file.close();

			target_in_args["root_path"] = target_path;
			target_in_args["optimization"]["enabled"] = true;
			state_reference.init(target_in_args, false, output_dir);
		}
		else
			throw std::runtime_error("Target json input missing!");

		logger().info("Start reference solve...");

		state_reference.load_mesh();

		// Mesh was not loaded successfully; load_mesh() logged the error.
		if (state_reference.mesh == nullptr)
		{
			// Cannot proceed without a mesh.
			return EXIT_FAILURE;
		}

		state_reference.compute_mesh_stats();
		state_reference.build_basis();

		if (state_reference.problem->is_time_dependent())
		{
			state_reference.output_dir = "target";
			std::filesystem::create_directories(state_reference.output_dir);
		}

		state_reference.assemble_rhs();
		state_reference.assemble_stiffness_mat();
		state_reference.solve_problem();

		if (!state_reference.problem->is_time_dependent())
			state_reference.save_vtu(state_reference.resolve_output_path("target.vtu"), 0.);

		logger().info("Reference solve done!");
	}

	state.load_mesh();

	if (state.is_contact_enabled() && !state.args["solver"]["contact"]["barrier_stiffness"].is_number())
	{
		logger().error("Not fixing the barrier stiffness in optimization!");
		return EXIT_FAILURE;
	}

	// Mesh was not loaded successfully; load_mesh() logged the error.
	if (state.mesh == nullptr)
	{
		// Cannot proceed without a mesh.
		return EXIT_FAILURE;
	}

	state.compute_mesh_stats();
	state.build_basis();

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	assert(state.formulation() == "LinearElasticity" || state.formulation() == "NeoHookean");
	std::shared_ptr<CompositeFunctional> func;
	if (objective_params["type"] == "trajectory")
	{
		if (matching_type == "exact")
			func = CompositeFunctional::create("Trajectory");
		else if (matching_type == "sdf")
			func = CompositeFunctional::create("SDFTrajectory");
		else if (matching_type == "exact-center")
			func = CompositeFunctional::create("CenterTrajectory");
		else if (matching_type == "last-center")
			func = CompositeFunctional::create("CenterXZTrajectory");
		else if (matching_type == "sine")
			func = CompositeFunctional::create("TargetY");
		else if (matching_type == "center-data")
			func = CompositeFunctional::create("CenterXYTrajectory");
		else if (matching_type == "marker-data")
			func = CompositeFunctional::create("NodeTrajectory");
		else
			logger().error("Invalid matching type!");
	}
	else if (objective_params["type"] == "height")
	{
		func = CompositeFunctional::create("Height");
	}

	if (transient_integral_type != "")
		func->set_transient_integral_type(transient_integral_type);

	std::set<int> interested_body_ids;
	const auto &interested_bodies = objective_params["volume_selection"].get<std::vector<int>>();
	interested_body_ids = std::set(interested_bodies.begin(), interested_bodies.end());

	std::set<int> interested_boundary_ids;
	const auto &interested_boundaries = objective_params["surface_selection"].get<std::vector<int>>();
	interested_boundary_ids = std::set(interested_boundaries.begin(), interested_boundaries.end());

	func->set_interested_ids(interested_body_ids, interested_boundary_ids);

	if (matching_type == "exact")
	{
		auto &f = *dynamic_cast<TrajectoryFunctional *>(func.get());

		std::set<int> reference_cached_body_ids;
		if (objective_params["type"] == "trajectory" && matching_type == "exact")
		{
			if (objective_params["reference_cached_body_ids"].size() > 0)
			{
				const auto &ref_cached = objective_params["reference_cached_body_ids"].get<std::vector<int>>();
				reference_cached_body_ids = std::set(ref_cached.begin(), ref_cached.end());
			}
			else
			{
				reference_cached_body_ids = interested_body_ids;
			}
		}

		f.set_reference(&state_reference, state, reference_cached_body_ids);
	}
	else if (matching_type == "sdf")
	{
		// TODO: Ingest this data from the json.
		auto &f = *dynamic_cast<SDFTrajectoryFunctional *>(func.get());
		if (control_points.size() == 0 || tangents.size() == 0)
		{
			control_points.setZero(2, 2);
			control_points << 1.2, -1.7,
				1.2, 1.7;
			tangents.setZero(2, 2);
			tangents << 2.5, 2,
				-1.0, 1;
		}
		if (delta.size() == 0)
		{
			delta.setZero(1, 2);
			delta << 0.01, 0.01;
		}
		logger().info("Control points are: {}", control_points);
		logger().info("Tangents are: {}", tangents);
		f.set_spline_target(control_points, tangents, delta);
		f.set_transient_integral_type("final");
	}
	else if (matching_type == "exact-center")
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
	else if (matching_type == "last-center")
	{
		auto &f = *dynamic_cast<CenterXZTrajectoryFunctional *>(func.get());
		f.set_transient_integral_type("final");
		std::vector<Eigen::VectorXd> barycenters(1);
		barycenters[0] = target_position;
		f.set_center_series(barycenters);
	}
	else if (matching_type == "sine")
	{
		auto &f = *dynamic_cast<TargetYFunctional *>(func.get());
		f.set_target_function([](const double x) {
			return sin(x) * 0.7;
		});
		f.set_target_function_derivative([](const double x) {
			return cos(x) * 0.7;
		});
	}
	else if (matching_type == "center-data")
	{
		auto &f = *dynamic_cast<CenterXYTrajectoryFunctional *>(func.get());
		std::ifstream infile(target_path);
		std::vector<Eigen::VectorXd> centers;
		double x = 0, y = 0;
		Eigen::VectorXd center;
		center.setZero(3);
		while (infile.good())
		{
			infile >> x;
			infile >> y;
			center(0) = x / 100;
			center(1) = y / 100; // cm to m
			centers.push_back(center);
		}
		infile.close();
		f.set_center_series(centers);
		f.set_transient_integral_type("uniform");
		print_centers(centers);
	}
	else if (matching_type == "marker-data")
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
		shape_optimization(state, func);
	else if (opt_type == "material")
		material_optimization(state, func);
	else if (opt_type == "initial")
		initial_condition_optimization(state, func);
	else if (opt_type == "control")
		control_optimization(state, func);
	else if (opt_type == "" && opt_types.size() > 0)
		general_optimization(state, func);
	else
		logger().error("Invalid optimization type!");

	return EXIT_SUCCESS;
}
