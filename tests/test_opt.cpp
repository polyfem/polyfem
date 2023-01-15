////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>

#include <iostream>
#include <fstream>
#include <catch2/catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace solver;
using namespace polysolve;

namespace
{
	bool load_json(const std::string &json_file, json &out)
	{
		std::ifstream file(json_file);

		if (!file.is_open())
			return false;

		file >> out;

		out["root_path"] = json_file;

		return true;
	}

	std::string resolve_output_path(const std::string &output_dir, const std::string &path)
	{
		if (std::filesystem::path(path).is_absolute())
			return path;
		else
			return std::filesystem::weakly_canonical(std::filesystem::path(output_dir) / path).string();
	}

	bool save_mat(const Eigen::MatrixXd &mat, const std::string &file_name)
	{
		std::ofstream file(file_name);
		if (!file.is_open())
			return false;

		file << fmt::format("matrix size {} x {}\n", mat.rows(), mat.cols());
		file << mat;

		return true;
	}

	std::vector<double> read_energy(const std::string &file)
	{
		std::ifstream energy_out(file);
		std::vector<double> energies;
		std::string line;
		if (energy_out.is_open())
		{
			while (getline(energy_out, line))
			{
				energies.push_back(std::stod(line.substr(0, line.find(","))));
			}
		}
		double starting_energy = energies[0];
		double optimized_energy = energies[energies.size() - 1];

		for (int i = 0; i < energies.size(); ++i)
		{
			if (i == 0)
				std::cout << "initial " << energies[i] << std::endl;
			else if (i == energies.size() - 1)
				std::cout << "final " << energies[i] << std::endl;
			else
				std::cout << "step " << i << " " << energies[i] << std::endl;
		}

		return energies;
	}

	void run_trajectory_opt(const std::string &name)
	{
		const std::string path = POLYFEM_DATA_DIR + std::string("/../optimizations/" + name);

		json target_args, in_args;
		load_json(path + "/run.json", in_args);
		load_json(path + "/target.json", target_args);
		auto state = create_state(in_args);
		auto target_state = create_state(target_args);
		solve_pde(*target_state);

		auto opt_params = state->args["optimization"];
		auto objective_params = opt_params["functionals"][0];

		std::string matching_type = objective_params["matching"];
		std::shared_ptr<CompositeFunctional> func;
		if (objective_params["type"] == "trajectory")
		{
			if (matching_type == "exact")
				func = CompositeFunctional::create("Trajectory");
			else if (matching_type == "sdf")
				func = CompositeFunctional::create("SDFTrajectory");
			else
				logger().error("Invalid matching type!");
		}
		else if (objective_params["type"] == "height")
		{
			func = CompositeFunctional::create("Height");
		}

		std::string transient_integral_type = objective_params["transient_integral_type"];
		if (transient_integral_type != "")
			func->set_transient_integral_type(transient_integral_type);

		std::set<int> interested_body_ids;
		std::vector<int> interested_bodies = objective_params["volume_selection"];
		interested_body_ids = std::set(interested_bodies.begin(), interested_bodies.end());

		std::set<int> interested_boundary_ids;
		std::vector<int> interested_boundaries = objective_params["surface_selection"];
		interested_boundary_ids = std::set(interested_boundaries.begin(), interested_boundaries.end());

		func->set_interested_ids(interested_body_ids, interested_boundary_ids);

		if (matching_type == "exact")
		{
			auto &f = *dynamic_cast<TrajectoryFunctional *>(func.get());

			std::set<int> reference_cached_body_ids;
			if (objective_params["reference_cached_body_ids"].size() > 0)
			{
				std::vector<int> ref_cached = objective_params["reference_cached_body_ids"];
				reference_cached_body_ids = std::set(ref_cached.begin(), ref_cached.end());
			}
			else
				reference_cached_body_ids = interested_body_ids;

			f.set_reference(target_state.get(), *state, reference_cached_body_ids);
		}

		CHECK_THROWS_WITH(general_optimization(*state, func), Catch::Matchers::Contains("Reached iteration limit"));
	}

	void run_opt_new(const std::string &name)
	{
		const std::string root_folder = POLYFEM_DATA_DIR + std::string("/../optimizations/") + name + "/";
		json opt_args;
		if (!load_json(resolve_output_path(root_folder, "run.json"), opt_args))
			log_and_throw_error("Failed to load optimization json file!");

		for (auto &state_arg : opt_args["states"])
			state_arg["path"] = resolve_output_path(root_folder, state_arg["path"]);

		auto nl_problem = make_nl_problem(opt_args, spdlog::level::level_enum::err);

		Eigen::VectorXd x = nl_problem->initial_guess();

		std::shared_ptr<cppoptlib::NonlinearSolver<solver::AdjointNLProblem>> nlsolver = make_nl_solver<solver::AdjointNLProblem>(opt_args["solver"]["nonlinear"]);

		CHECK_THROWS_WITH(nlsolver->minimize(*nl_problem, x), Catch::Matchers::Contains("Reached iteration limit"));
	}
} // namespace

#if defined(__linux__)

TEST_CASE("material-opt", "[optimization]")
{
	run_opt_new("material-opt");
	auto energies = read_energy("material-opt");

	REQUIRE(energies[0] == Approx(5.95421809553).epsilon(1e-3));
	REQUIRE(energies[energies.size() - 1] == Approx(0.00101793422213).epsilon(1e-3));
}

TEST_CASE("friction-opt", "[optimization]")
{
	run_opt_new("friction-opt");
	auto energies = read_energy("friction-opt");

	REQUIRE(energies[0] == Approx(0.000103767819516).epsilon(1e-1));
	REQUIRE(energies[energies.size() - 1] == Approx(3.26161994783e-07).epsilon(1e-1));
}

TEST_CASE("damping-opt", "[optimization]")
{
	run_opt_new("damping-opt");
	auto energies = read_energy("damping-opt");

	REQUIRE(energies[0] == Approx(4.14517346014e-07).epsilon(1e-3));
	REQUIRE(energies[energies.size() - 1] == Approx(2.12684299792e-09).epsilon(1e-3));
}

TEST_CASE("initial-opt", "[optimization]")
{
	run_trajectory_opt("initial-opt");
	auto energies = read_energy("initial-opt");

	REQUIRE(energies[0] == Approx(0.147092).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(0.109971).epsilon(1e-4));
}

TEST_CASE("topology-opt", "[optimization]")
{
	run_opt_new("topology-opt");
	auto energies = read_energy("topology-opt");

	REQUIRE(energies[0] == Approx(136.014).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(1.73135).epsilon(1e-4));
}

TEST_CASE("shape-stress-opt-new", "[optimization]")
{
	run_opt_new("shape-stress-opt-new");
	auto energies = read_energy("shape-stress-opt-new");

	REQUIRE(energies[0] == Approx(12.0735).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(11.3886).epsilon(1e-4));
}

TEST_CASE("shape-trajectory-surface-opt-new", "[optimization]")
{
	run_opt_new("shape-trajectory-surface-opt-new");
	auto energies = read_energy("shape-trajectory-surface-opt-new");

	REQUIRE(energies[0] == Approx(6.1658e-05).epsilon(1e-3));
	REQUIRE(energies[energies.size() - 1] == Approx(3.6194e-05).epsilon(1e-3));
}

TEST_CASE("shape-trajectory-surface-opt-bspline", "[optimization]")
{
	run_opt_new("shape-trajectory-surface-opt-bspline");
	auto energies = read_energy("shape-trajectory-surface-opt-bspline");

	REQUIRE(energies[0] == Approx(6.1658e-05).epsilon(1e-3));
	REQUIRE(energies[energies.size() - 1] == Approx(3.6194e-05).epsilon(1e-3));
}

TEST_CASE("multiparameter-sdf-trajectory-surface-opt", "[optimization]")
{
	run_opt_new("multiparameter-sdf-trajectory-surface-opt");
	auto energies = read_energy("multiparameter-sdf-trajectory-surface-opt");

	REQUIRE(energies[0] == Approx(0.15327).epsilon(1e-3));
	REQUIRE(energies[energies.size() - 1] == Approx(0.11259).epsilon(1e-3));
}

// TEST_CASE("sdf-test", "[optimization]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/../optimizations/multiparameter-sdf-trajectory-surface-opt");

// 	json in_args;
// 	load_json(path + "/state.json", in_args);
// 	auto state = create_state(in_args);
// 	json args = R"(
// 		{
// 			"surface_selection": [3, 4]
// 		}
// 	)"_json;
// 	SDFTargetObjective sdf(*state, nullptr, args);
// 	Eigen::MatrixXd control_points(4, 2);
// 	control_points << 0, 1,
// 		0.5, 0.7,
// 		0.5, 0.3,
// 		0, 0;
// 	Eigen::VectorXd knots(8);
// 	knots << 0, 0, 0, 0, 1, 1, 1, 1;
// 	Eigen::MatrixXd delta(2, 1);
// 	delta << 0.1, 0.1;
// 	sdf.set_bspline_target(control_points, knots, delta(0));

// 	int sampling = (int)(3 / delta(0));
// 	int upsampling = 1000;
// 	Eigen::MatrixXd distance(sampling, sampling);
// 	Eigen::MatrixXd grad_x;
// 	Eigen::MatrixXd grad_y;
// 	grad_x.setZero(sampling, sampling);
// 	grad_y.setZero(sampling, sampling);
// 	Eigen::MatrixXd bounds(2, 2);
// 	bounds << -1, 2,
// 		-1, 2;
// 	for (int i = 0; i < sampling; ++i)
// 		for (int j = 0; j < sampling; ++j)
// 		{
// 			double x = bounds(0, 0) + j * (bounds(0, 1) - bounds(0, 0)) / (double)sampling;
// 			double y = bounds(1, 1) - i * (bounds(1, 1) - bounds(1, 0)) / (double)sampling;
// 			Eigen::MatrixXd point(2, 1);
// 			point << x, y;
// 			double d;
// 			Eigen::MatrixXd g;
// 			sdf.compute_distance(point, d);
// 			distance(i, j) = d;
// 			grad_x(i, j) = 0;
// 			grad_y(i, j) = 0;
// 		}
// 	for (int i = 1; i < sampling - 1; ++i)
// 		for (int j = 1; j < sampling - 1; ++j)
// 		{
// 			grad_x(i, j) = (1. / 2. / delta(0)) * (distance(i, j + 1) - distance(i, j - 1));
// 			grad_y(i, j) = (1. / 2. / delta(0)) * (distance(i - 1, j) - distance(i + 1, j));
// 		}

// 	save_mat(distance, "orig_distance.txt");
// 	save_mat(grad_x, "orig_grad_x.txt");
// 	save_mat(grad_y, "orig_grad_y.txt");

// 	distance.resize(upsampling, upsampling);
// 	grad_x.resize(upsampling, upsampling);
// 	grad_y.resize(upsampling, upsampling);
// 	for (int i = 0; i < upsampling; ++i)
// 		for (int j = 0; j < upsampling; ++j)
// 		{
// 			double x = bounds(0, 0) + j * (bounds(0, 1) - bounds(0, 0)) / (double)upsampling;
// 			double y = bounds(1, 1) - i * (bounds(1, 1) - bounds(1, 0)) / (double)upsampling;
// 			Eigen::MatrixXd point(2, 1);
// 			point << x, y;
// 			double d;
// 			Eigen::MatrixXd g;
// 			sdf.evaluate(point, d, g);
// 			// sdf.compute_distance(point, d);
// 			distance(i, j) = d;
// 			grad_x(i, j) = g(0);
// 			grad_y(i, j) = g(1);
// 		}

// 	save_mat(distance, "distance.txt");
// 	save_mat(grad_x, "grad_x.txt");
// 	save_mat(grad_y, "grad_y.txt");
// }

TEST_CASE("3d-bspline-shape-trajectory-opt", "[optimization]")
{
	run_opt_new("3d-bspline-shape-trajectory-opt");
	auto energies = read_energy("3d-bspline-shape-trajectory-opt");

	REQUIRE(energies[0] == Approx(0.00473695).epsilon(1e-3));
	REQUIRE(energies[energies.size() - 1] == Approx(0.000461251).epsilon(1e-4));
}

TEST_CASE("3d-bspline-shape-matching", "[optimization]")
{
	run_opt_new("3d-bspline-shape-matching");
	auto energies = read_energy("3d-bspline-shape-matching");

	REQUIRE(energies[0] == Approx(1.86898e-05).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(1.85359e-05).epsilon(1e-4));
}

#endif