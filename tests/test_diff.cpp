#include <polyfem/legacy/State.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/optimization/Optimizations.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/BuildFromJson.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/OptState.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <polyfem/optimization/var2sims/ElasticVariableToSimulation.hpp>
#include <polyfem/optimization/forms/SmoothingForms.hpp>
#include <polyfem/optimization/forms/TargetForms.hpp>
#include <polyfem/optimization/parametrization/Parametrizations.hpp>

#include <finitediff.hpp>
#include <catch2/catch_all.hpp>

#include <cmath>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <string_view>

using namespace polyfem;
using namespace solver;

// Test the accurarcy of objective functional gradient.
//
// Tips: To debug regression, set PRINT_DEBUG_LOG = true and use -V flag in ctest.

// Some tests is computationally expensive.
// To avoid destroying CI time, we only test those in release build.
#ifdef NDEBUG
#define EXPENSIVE_TEST_LABEL "[opt_gradient]"
#else
#define EXPENSIVE_TEST_LABEL "[.][opt_gradient]"
#endif

namespace
{
	constexpr uint64_t BASE_SEED = 0x6a09e667f3bcc909ULL;
	constexpr bool PRINT_DEBUG_LOG = false; // for debugging

	std::string append_root_path(std::string_view path)
	{
		return POLYFEM_DIFF_DIR + std::string("/input/") + std::string(path);
	}

	/// @brief Load config json and patch root_path.
	bool load_json(const std::string &json_file, json &out)
	{
		std::ifstream file(json_file);

		if (!file.is_open())
		{
			return false;
		}
		file >> out;

		out["root_path"] = json_file;

		return true;
	}

	void sample_field(const legacy::State &state, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> field, Eigen::MatrixXd &discrete_field, const int order = 1)
	{
		Eigen::MatrixXd tmp;
		tmp.setZero(1, state.mesh->dimension());
		tmp = field(tmp);
		const int actual_dim = tmp.cols();

		if (order >= 1)
		{
			Eigen::MatrixXd V;
			state.get_vertices(V);

			discrete_field = utils::flatten(field(V));
		}
		else if (order == 0)
		{
			Eigen::MatrixXd centers;
			if (state.mesh->is_volume())
				state.mesh->cell_barycenters(centers);
			else
				state.mesh->face_barycenters(centers);

			discrete_field = utils::flatten(field(centers));
		}
	}

	// for debugging. compute the relative tolerance that will pass adjoint test.
	double compute_relative_eps(double derivative, double finite_difference)
	{
		double denom = std::max(1e-30, std::abs(finite_difference));
		return std::abs(derivative - finite_difference) / denom;
	}

	/// @brief Sample random matrix in range [min, max].
	///
	/// We do not use Eigen built-in Random because it is not stable across version.
	Eigen::MatrixXd uniform_random_matrix(
		int rows,
		int cols,
		std::mt19937_64 &rng,
		double min_value,
		double max_value)
	{
		std::uniform_real_distribution<double> dist(min_value, max_value);
		return Eigen::MatrixXd::NullaryExpr(rows, cols, [&]() { return dist(rng); });
	}

	/// @brief Verify objective functional gradient against a small perturbation.
	/// @param problem AdjointNLProblem.
	/// @param x Solution.
	/// @param theta Perturbation direction.
	/// @param dt Perturbation scale.
	/// @param tol Gradient tolerance.
	/// @param test_tag Debug info.
	/// @param trial Debug info.
	/// @param seed Debug info.
	void verify_adjoint(
		AdjointNLProblem &problem,
		const Eigen::VectorXd &x,
		const Eigen::MatrixXd &theta,
		double dt,
		double tol,
		std::string_view test_tag = "",
		int trial = 0,
		uint64_t seed = 0)
	{
		problem.solution_changed(x);
		// problem.save_to_file(0, x); // debug only
		double functional_val = problem.value(x);

		Eigen::VectorXd one_form;
		problem.gradient(x, one_form);
		double derivative = (one_form.array() * theta.array()).sum();

		problem.solution_changed(x + theta * dt);
		double next_functional_val = problem.value(x + theta * dt);

		problem.solution_changed(x - theta * dt);
		double former_functional_val = problem.value(x - theta * dt);

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		double back_finite_difference = (functional_val - former_functional_val) / dt;
		double front_finite_difference = (next_functional_val - functional_val) / dt;
		if (PRINT_DEBUG_LOG)
		{
			std::string prefix = fmt::format("[adjoint test debugging] {} ({}):", test_tag, trial);
			logger().info("{} f(x) {:.16f} f(x-dt) {:.16f} f(x+dt) {:.16f}", prefix, functional_val, former_functional_val, next_functional_val);
			logger().info("{} forward fd {:.16f} backward fd {:.16f}", prefix, front_finite_difference, back_finite_difference);
			logger().info("{} derivative: {:.12f} fd: {:.12f}", prefix, derivative, finite_difference);
			double eps = compute_relative_eps(derivative, finite_difference);
			logger().info("{} relative error (vs fd): {:.12f}", prefix, eps);
		}
		REQUIRE(derivative == Catch::Approx(finite_difference).epsilon(tol));
	}

	class TestContext
	{
	public:
		json args;
		OptState opt;

		TestContext(std::string_view json_name)
		{
			load_json(append_root_path(json_name), args);

			// Silence polyFEM logging.
			args["output"]["directory"] = "";
			args["output"]["save_frequency"] = 100000;
			args["output"]["log"]["path"] = "";
			args["output"]["log"]["level"] = "off";
			args["output"]["log"]["file_level"] = "off";
			args["output"]["log"]["quiet"] = true;

			// Disable multi-threading.
			// Due to floating point parallel reduction multi-threading will introduce randomess to the test.
			// Which might break carefully tuned tolerance.
			args["solver"]["max_threads"] = 1;

			opt.init(args, false);
			opt.create_states(/*max_threads=*/1);
			opt.init_variables();
			opt.create_problem();
		}
	};

	/// @brief Run optimization test variant 1.
	///
	/// 1. Build test context.
	/// 2. Compute initial guess via inverse_evaluation.
	/// 3. Gen random velocity.
	/// 4. Check convergence through verify_adjoint.
	///
	/// @param[in] json Filename of the input json config.
	/// @param[in] dt Time step.
	/// @param[in] tol Convergence check tolerance.
	/// @param[in] rand_min Minimum random value.
	/// @param[in] rand_max Maximum random value.
	/// @param[in] seed Random seed.
	/// @param[in] repeat Repeat number for gradient test.
	void run_test1(
		std::string_view json_name,
		float dt,
		float tol,
		double rand_min,
		double rand_max,
		uint64_t seed,
		int repeat)
	{
		TestContext ctx{json_name};

		Eigen::VectorXd x;
		ctx.opt.initial_guess(x);

		std::mt19937_64 rng(seed);
		for (int i = 0; i < repeat; ++i)
		{
			Eigen::MatrixXd velocity = uniform_random_matrix(x.size(), 1, rng, rand_min, rand_max);
			verify_adjoint(*ctx.opt.nl_problem, x, velocity, dt, tol, json_name, i, seed);
		}
	}

	/// @brief Run optimization test variant 2.
	///
	/// Assume only a single state exists.
	/// 1. Build test context.
	/// 2. Compute initial guess via inverse_eval.
	/// 3. Set gradient as velocity.
	/// 4. Check convergence through verify_adjoint.
	///
	/// @param[in] json Filename of the input json config.
	/// @param[in] dt Time step.
	/// @param[in] tol Convergence check tolerance.
	/// @param[in] seed Random seed.
	/// @param[in] repeat Repeat number for gradient test.
	void run_test2(
		std::string_view json_name,
		float dt,
		float tol,
		uint64_t seed,
		int repeat)
	{
		TestContext ctx{json_name};

		Eigen::VectorXd x;
		ctx.opt.initial_guess(x);

		ctx.opt.nl_problem->solution_changed(x);
		Eigen::VectorXd one_form;
		ctx.opt.nl_problem->gradient(x, one_form);

		for (int i = 0; i < repeat; ++i)
		{
			verify_adjoint(*ctx.opt.nl_problem, x, one_form.normalized(), dt, tol, json_name, i, seed);
		}
	}

} // namespace

TEST_CASE("laplacian", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 0;
	constexpr int REPEAT = 1;
	constexpr double TOL = 8.3e-7;
	TestContext ctx{"laplacian-opt.json"};

	Eigen::VectorXd x;
	ctx.opt.initial_guess(x);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = vel(i) * std::cos(vel(i));
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	sample_field(*(ctx.opt.states[0]), velocity, velocity_discrete);

	for (int i = 0; i < REPEAT; ++i)
	{
		verify_adjoint(*ctx.opt.nl_problem, x, velocity_discrete, 1e-7, TOL, "laplacian", i, SEED);
	}
}

TEST_CASE("linear_elasticity-surface-3d", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 1;
	constexpr int REPEAT = 3;
	constexpr double TOL = 3.0e-7;
	run_test1("linear_elasticity-surface-3d-opt.json", 1e-7, TOL, -1.0, 1.0, SEED, REPEAT);
}

TEST_CASE("linear_elasticity-surface", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 2;
	constexpr int REPEAT = 3;
	constexpr double TOL = 2.3e-7;
	run_test1("linear_elasticity-surface-opt.json", 1e-6, TOL, -1.0, 1.0, SEED, REPEAT);
}

TEST_CASE("topology-compliance", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 3;
	constexpr int REPEAT = 3;
	constexpr double TOL = 2.5e-5;
	run_test1("topology-compliance-opt.json", 1e-2, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("neohookean-stress-3d", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 4;
	constexpr int REPEAT = 3;
	constexpr double TOL = 3.7e-5;
	run_test1("neohookean-stress-3d-opt.json", 1e-7, TOL, -1.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-neumann-nodes", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 5;
	constexpr int REPEAT = 3;
	constexpr double TOL = 2.0e-5;
	run_test1("shape-neumann-nodes-opt.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-pressure-nodes-2d", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 6;
	constexpr int REPEAT = 3;
	constexpr double TOL = 5.5e-3;
	run_test1("shape-pressure-nodes-2d-opt.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("static-control-pressure-nodes-3d", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 7;
	constexpr int REPEAT = 2;
	constexpr double TOL = 2.7e-7;
	run_test1("static-control-pressure-nodes-3d-opt.json", 1e-3, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("control-pressure-walker-2d", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 8;
	constexpr int REPEAT = 3;
	constexpr double TOL = 1.8e-5;
	run_test1("walker-opt.json", 1e-4, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-walker-2d", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 9;
	constexpr int REPEAT = 3;
	constexpr double TOL = 2.2e-4;
	run_test1("walker-shape-opt.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-contact-force-norm", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 10;
	constexpr int REPEAT = 3;
	constexpr double TOL = 1.3e-6;
	run_test1("shape-contact-force-norm-opt.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-contact-force-norm-adhesion", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 11;
	constexpr int REPEAT = 3;
	constexpr double TOL = 5.3e-2;
	run_test1("shape-contact-force-norm-opt-adhesion.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-contact-force-norm-3d", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 12;
	constexpr int REPEAT = 3;
	constexpr double TOL = 2.3e-7;
	run_test1("shape-contact-force-norm-3d-opt.json", 1e-6, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-contact", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 27;
	constexpr int REPEAT = 1;
	constexpr double TOL = 7.0e-7;
	run_test2("shape-contact-opt.json", 1e-7, TOL, SEED, REPEAT);
}

TEST_CASE("shape-contact-adhesion", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 28;
	constexpr int REPEAT = 1;
	constexpr double TOL = 2.4e-3;
	run_test2("shape-contact-adhesion-opt.json", 1e-7, TOL, SEED, REPEAT);
}

TEST_CASE("node-trajectory", "[opt_gradient]")
{
	// Prepare test manually because we need random target form.
	// The opt json is mostly a dummy json.

	json opt_args;
	load_json(append_root_path("node-trajectory-opt.json"), opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);
	opt_args["output"]["log"]["path"] = "";
	opt_args["output"]["log"]["level"] = "off";
	opt_args["output"]["log"]["file_level"] = "off";
	opt_args["output"]["log"]["quiet"] = true;

	// One state only.
	std::string root = POLYFEM_DIFF_DIR + std::string("/input/");
	auto states =
		from_json::build_states(root, opt_args["states"], -1, opt_args["output"]["log"]);
	std::vector<std::shared_ptr<DiffCache>> diff_caches = {std::make_shared<DiffCache>()};

	auto elastic_var2sim =
		std::make_shared<ElasticVariableToSimulation>(states, diff_caches, CompositeParametrization{});
	VariableToSimulationGroup var2sim_group;
	var2sim_group.data.push_back(elastic_var2sim);

	constexpr uint64_t SEED = BASE_SEED + 22;
	constexpr int REPEAT = 3;
	std::mt19937_64 rng(SEED);
	Eigen::MatrixXd targets =
		uniform_random_matrix(states[0]->n_bases, states[0]->mesh->dimension(), rng, 0.0, 10.0);

	// All active.
	std::vector<int> actives(targets.rows());
	for (int i = 0; i < targets.rows(); ++i)
	{
		actives[i] = i;
	}

	auto form =
		std::make_shared<NodeTargetForm>(states[0], diff_caches[0], var2sim_group, actives, targets);
	AdjointNLProblem problem{form, var2sim_group, states, diff_caches, opt_args};
	Eigen::VectorXd x = var2sim_group.data[0]->inverse_eval();
	constexpr double TOL = 4.9e-5;
	for (int i = 0; i < REPEAT; ++i)
	{
		Eigen::MatrixXd velocity = uniform_random_matrix(x.size(), 1, rng, 0.0, 1.0);
		verify_adjoint(problem, x, velocity, 1e-5, TOL, "node-trajectory", i, SEED);
	}
}

TEST_CASE("damping-transient", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 13;
	constexpr int REPEAT = 3;
	constexpr double TOL = 2.5e-6;
	run_test1("damping-transient-opt.json", 1e-3, TOL, 1.0, 1.0, SEED, REPEAT);
}

TEST_CASE("material-transient", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 14;
	constexpr int REPEAT = 3;
	constexpr double TOL = 5.3e-6;
	run_test1("material-transient-opt.json", 1e-5, TOL, 1e3, 1e3, SEED, REPEAT);
}

TEST_CASE("shape-transient-friction", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 15;
	constexpr int REPEAT = 3;
	constexpr double TOL = 1.1e-5;
	run_test1("shape-transient-friction-opt.json", 1e-6, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-transient-friction-sdf", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 16;
	constexpr int REPEAT = 3;
	constexpr double TOL = 2.7e-6;
	run_test1("shape-transient-friction-sdf-opt.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("shape-transient-stress-3d-frictionless-fast", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 31;
	constexpr int REPEAT = 1;
	constexpr double TOL = 1e-3;
	run_test2("shape-transient-stress-3d-frictionless-fast-opt.json", 1e-5, TOL, SEED, REPEAT);
}

TEST_CASE("3d-shape-mesh-target", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 17;
	constexpr int REPEAT = 1;
	constexpr double TOL = 6.2e-7;
	run_test1("3d-shape-mesh-target-opt.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("initial-contact-min-dist", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 18;
	constexpr int REPEAT = 3;
	constexpr double TOL = 1.9e-6;
	run_test1("initial-contact-min-dist-opt.json", 1e-5, TOL, -1.0, 1.0, SEED, REPEAT);
}

TEST_CASE("friction-contact", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 29;
	constexpr int REPEAT = 1;
	constexpr double TOL = 3.9e-7;
	TestContext ctx{"friction-contact-opt.json"};

	Eigen::VectorXd velocity = Eigen::VectorXd::Ones(1);
	Eigen::VectorXd x = 0.2f * Eigen::VectorXd::Ones(1);

	for (int i = 0; i < REPEAT; ++i)
	{
		verify_adjoint(*ctx.opt.nl_problem, x, velocity, 1e-4, TOL, "friction-contact", i, SEED);
	}
}

TEST_CASE("barycenter", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 30;
	constexpr int REPEAT = 1;
	constexpr double TOL = 8.1e-7;
	TestContext ctx{"barycenter-opt.json"};

	Eigen::VectorXd x;
	ctx.opt.initial_guess(x);

	int dof_num = ctx.opt.states[0]->ndof();
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Zero(ctx.opt.ndof, 1);
	for (int i = 0; i < ctx.opt.states[0]->n_bases; i++)
	{
		velocity(dof_num + i * 2 + 0) = -2.0f;
		velocity(dof_num + i * 2 + 1) = -1.0f;
	}

	for (int i = 0; i < REPEAT; ++i)
	{
		verify_adjoint(*ctx.opt.nl_problem, x, velocity, 1e-6, TOL, "barycenter", i, SEED);
	}
}

TEST_CASE("shape-contact-smooth", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 31;
	constexpr int REPEAT = 1;
	constexpr double TOL = 7.5e-7;
	TestContext ctx{"shape-contact-opt.json"};

	// Because state configs are shared, tailor json args.
	for (auto &state : ctx.opt.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["use_convergent_formulation"] = false;
		state->args["contact"]["alpha_t"] = 0.95;
	}

	Eigen::MatrixXd V;
	ctx.opt.states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	ctx.opt.nl_problem->solution_changed(x);
	Eigen::VectorXd one_form;
	ctx.opt.nl_problem->gradient(x, one_form);

	for (int i = 0; i < REPEAT; ++i)
	{
		verify_adjoint(*ctx.opt.nl_problem, x, one_form.normalized(), 1e-6, TOL, "shape-contact-smooth", i, SEED);
	}
}

TEST_CASE("initial-contact-smooth", "[opt_gradient]")
{
	TestContext ctx{"initial-contact-smooth-opt.json"};

	// Because state configs are shared, tailor json args.
	for (auto &state : ctx.opt.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["use_convergent_formulation"] = false;
		state->args["contact"]["alpha_t"] = 0.95;
		state->args["contact"]["friction_coefficient"] = 0;
	}

	Eigen::VectorXd x;
	ctx.opt.initial_guess(x);

	constexpr uint64_t SEED = BASE_SEED + 23;
	constexpr int REPEAT = 3;
	constexpr double TOL = 6.0e-7;
	std::mt19937_64 rng(SEED);
	for (int i = 0; i < REPEAT; ++i)
	{
		Eigen::MatrixXd velocity = uniform_random_matrix(ctx.opt.ndof, 1, rng, 0.0, 1.0);
		verify_adjoint(*ctx.opt.nl_problem, x, velocity, 1e-6, TOL, "initial-contact-smooth", i, SEED);
	}
}

TEST_CASE("shape-transient-smooth", EXPENSIVE_TEST_LABEL)
{
	TestContext ctx{"shape-transient-friction-opt.json"};

	// Because states are shared, tailor json args.
	for (auto &state : ctx.opt.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["alpha_t"] = 0.95;
		state->args["contact"]["friction_coefficient"] = 0;
		state->args["solver"]["nonlinear"]["grad_norm_tol"] = 1e-8;
	}

	Eigen::MatrixXd V;
	ctx.opt.states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	constexpr uint64_t SEED = BASE_SEED + 24;
	constexpr int REPEAT = 3;
	constexpr double TOL = 4.9e-6;
	std::mt19937_64 rng(SEED);
	for (int i = 0; i < REPEAT; ++i)
	{
		Eigen::MatrixXd velocity = uniform_random_matrix(ctx.opt.ndof, 1, rng, 0.0, 1.0);
		verify_adjoint(*ctx.opt.nl_problem, x, velocity, 1e-6, TOL, "shape-transient-smooth", i, SEED);
	}
}

TEST_CASE("shape-pressure-nodes-3d", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 19;
	constexpr int REPEAT = 1;
	constexpr double TOL = 4.5e-5;
	run_test1("shape-pressure-nodes-3d-opt.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("control-pressure-nodes-3d", EXPENSIVE_TEST_LABEL)
{
	constexpr uint64_t SEED = BASE_SEED + 20;
	constexpr int REPEAT = 1;
	constexpr double TOL = 1.8e-6;
	run_test1("control-pressure-nodes-3d-opt.json", 1e-8, TOL, 0.0, 1.0, SEED, REPEAT);
}

TEST_CASE("dirichlet-nodes-3d", "[opt_gradient]")
{
	constexpr uint64_t SEED = BASE_SEED + 21;
	constexpr int REPEAT = 3;
	constexpr double TOL = 1.8e-7;
	run_test1("dirichlet-nodes-3d-opt.json", 1e-7, TOL, 0.0, 1.0, SEED, REPEAT);
}

// Only on windows debug build homogenize tests failed with:
// "Failed to factorize constraints matrix"
//
// The old comment mentioned this is a cholmod problem.
#ifndef _WIN32

TEST_CASE("homogenize-stress-periodic", EXPENSIVE_TEST_LABEL)
{
	TestContext ctx{"homogenize-stress-periodic-opt.json"};

	Eigen::VectorXd x;
	ctx.opt.initial_guess(x);

	auto &state = *(ctx.opt.states[0]);
	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd min = V.colwise().minCoeff();
	Eigen::VectorXd max = V.colwise().maxCoeff();

	constexpr uint64_t SEED = BASE_SEED + 25;
	constexpr int REPEAT = 3;
	constexpr double TOL = 2.3e-7;
	std::mt19937_64 rng(SEED);
	constexpr double BOUNDARY_EPS = 1e-5;

	for (int trial = 0; trial < REPEAT; ++trial)
	{
		Eigen::MatrixXd velocity = uniform_random_matrix(x.size(), 1, rng, 0.0, 1.0);
		for (int i = 0; i < V.rows(); i++)
		{
			auto vert = state.mesh->point(i);
			if (vert(0) < min(0) + BOUNDARY_EPS || vert(0) > max(0) - BOUNDARY_EPS || vert(1) < min(1) + BOUNDARY_EPS || vert(1) > max(1) - BOUNDARY_EPS)
			{
				for (int d = 0; d < 2; d++)
					velocity(i * 2 + d) = 0;
			}
		}
		verify_adjoint(*ctx.opt.nl_problem, x, velocity, 1e-7, TOL, "homogenize-stress-periodic", trial, SEED);
	}
}

TEST_CASE("homogenize-stress", EXPENSIVE_TEST_LABEL)
{
	TestContext ctx{"homogenize-stress-opt.json"};

	Eigen::VectorXd x;
	ctx.opt.initial_guess(x);

	auto &state = *(ctx.opt.states[0]);
	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd min = V.colwise().minCoeff();
	Eigen::VectorXd max = V.colwise().maxCoeff();

	constexpr uint64_t SEED = BASE_SEED + 26;
	constexpr int REPEAT = 3;
	constexpr double TOL = 7.5e-7;
	std::mt19937_64 rng(SEED);
	constexpr double BOUNDARY_EPS = 1e-5;

	for (int trial = 0; trial < REPEAT; ++trial)
	{
		Eigen::MatrixXd velocity = uniform_random_matrix(x.size(), 1, rng, 0.0, 1.0);
		for (int i = 0; i < V.rows(); i++)
		{
			auto vert = state.mesh->point(i);
			if (vert(0) < min(0) + BOUNDARY_EPS || vert(0) > max(0) - BOUNDARY_EPS || vert(1) < min(1) + BOUNDARY_EPS || vert(1) > max(1) - BOUNDARY_EPS)
			{
				for (int d = 0; d < 2; d++)
					velocity(i * 2 + d) = 0;
			}
		}
		verify_adjoint(*ctx.opt.nl_problem, x, velocity, 1e-7, TOL, "homogenize-stress", trial, SEED);
	}
}

#endif
