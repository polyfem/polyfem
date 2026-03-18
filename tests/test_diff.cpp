#include <polyfem/State.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/optimization/Optimizations.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/BuildFromJson.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <polyfem/optimization/forms/SmoothingForms.hpp>
#include <polyfem/optimization/forms/TargetForms.hpp>
#include <polyfem/optimization/parametrization/Parametrizations.hpp>
#include <polyfem/optimization/parametrization/NodeCompositeParametrizations.hpp>

#include <finitediff.hpp>
#include <catch2/catch_all.hpp>

#include <cmath>
#include <fstream>
#include <cstdlib>

// Tests here are *unstable*.
//
// There are two levels of randomness:
// 1. We generate radom perturbation (velocity) using Catch2 seed.
// 2. Polyfem itself is non-deterministic. Even test with the same
//    rng seed might fail randomly. Based on my test, non-deterministic
//    behavior only happens when multi-threading is enabled.
//    So I am guessing the culprit is non-associative floating point reduction.
//
// If you encouter test failure here don't panic. Try:
// 1. First fix the perturbation by specifying --rng-seed
// 2. If still non-deterministic, set max thread to 1.
// 3. Then decide if this is an actual regression.

using namespace polyfem;
using namespace solver;

namespace
{
	std::string append_root_path(const std::string &path)
	{
		return POLYFEM_DIFF_DIR + std::string("/input/") + path;
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

	void sample_field(const State &state, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> field, Eigen::MatrixXd &discrete_field, const int order = 1)
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

	Eigen::MatrixXd unflatten(const Eigen::VectorXd &x, int dim)
	{
		if (x.size() == 0)
			return Eigen::MatrixXd(0, dim);

		assert(x.size() % dim == 0);
		Eigen::MatrixXd X(x.size() / dim, dim);
		for (int i = 0; i < x.size(); ++i)
		{
			X(i / dim, i % dim) = x(i);
		}
		return X;
	}

	void verify_adjoint(AdjointNLProblem &problem, const Eigen::VectorXd &x, const Eigen::MatrixXd &theta, const double dt, const double tol)
	{
		problem.solution_changed(x);
		problem.save_to_file(0, x);
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
		logger().trace("f(x) {:.16f} f(x-dt) {:.16f} f(x+dt) {:.16f}", functional_val, former_functional_val, next_functional_val);
		logger().trace("forward fd {:.16f} backward fd {:.16f}", front_finite_difference, back_finite_difference);
		logger().trace("derivative: {:.12f} fd: {:.12f}", derivative, finite_difference);
		logger().trace("relative error: {:.12f}", abs((finite_difference - derivative) / derivative));
		REQUIRE(derivative == Catch::Approx(finite_difference).epsilon(tol));
	}

	class TestContext
	{
	public:
		json args;
		std::vector<std::shared_ptr<State>> states;
		std::vector<std::shared_ptr<DiffCache>> diff_caches;
		int ndof;
		std::vector<int> var_sizes;
		std::shared_ptr<AdjointForm> form;
		VariableToSimulationGroup var2sim;
		// Use unique_ptr here because it has no default ctor by design.
		std::unique_ptr<AdjointNLProblem> problem;

		TestContext(const std::string &json_name)
		{
			load_json(append_root_path(json_name), args);
			args = AdjointOptUtils::apply_opt_json_spec(args, false);

			// Build states.
			std::string root = POLYFEM_DIFF_DIR + std::string("/input/");
			states = from_json::build_states(root, args["states"], -1);

			// Build diff_caches.
			diff_caches.resize(states.size());
			for (auto &cache : diff_caches)
			{
				cache = std::make_shared<DiffCache>();
			}

			// Build dof and variable sizes.
			ndof = 0;
			for (const auto &arg : args["parameters"])
			{
				int size = AdjointOptUtils::compute_variable_size(arg, states);
				ndof += size;
				var_sizes.push_back(size);
			}

			// Build variable to simulation.
			var2sim =
				from_json::build_variable_to_simulation_group(args["variable_to_simulation"], states, diff_caches, var_sizes);

			// Build forms.
			form = from_json::build_form(
				args["functionals"], var2sim, states, diff_caches);

			// Build adjoint NL problem.
			problem = std::make_unique<AdjointNLProblem>(form, var2sim, states, diff_caches, args);
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
	void run_test1(
		const std::string &json_name,
		float dt,
		float tol,
		double rand_min,
		double rand_max)
	{
		TestContext ctx{json_name};

		// Compute initial solution.
		Eigen::VectorXd x =
			AdjointOptUtils::inverse_evaluation(ctx.args["parameters"], ctx.ndof, ctx.var_sizes, ctx.var2sim);

		// Gen random velocity using reproducible Catch2 seed.
		std::srand(Catch::rngSeed());
		Eigen::MatrixXd velocity =
			((Eigen::MatrixXd::Random(x.size(), 1).array() + 1.0) * 0.5) * (rand_max - rand_min) + rand_min;

		verify_adjoint(*ctx.problem, x, velocity, dt, tol);
	}

	/// @brief Run optimization test variant 2.
	///
	/// Assume only a single state exists.
	/// 1. Build test context.
	/// 2. Use vertex position as initial guess.
	/// 3. Gen random velocity.
	/// 4. Check convergence through verify_adjoint.
	///
	/// @param[in] json Filename of the input json config.
	/// @param[in] dt Time step.
	/// @param[in] tol Convergence check tolerance.
	/// @param[in] rand_min Minimum random value.
	/// @param[in] rand_max Maximum random value.
	void run_test2(
		const std::string &json_name,
		float dt,
		float tol,
		double rand_min,
		double rand_max)
	{
		TestContext ctx{json_name};

		// Compute initial solution.
		Eigen::MatrixXd V;
		ctx.states[0]->get_vertices(V);
		Eigen::VectorXd x = utils::flatten(V);

		// Gen random velocity using reproducible Catch2 seed.
		std::srand(Catch::rngSeed());
		Eigen::MatrixXd velocity =
			((Eigen::MatrixXd::Random(x.size(), 1).array() + 1.0) * 0.5) * (rand_max - rand_min) + rand_min;

		verify_adjoint(*ctx.problem, x, velocity, dt, tol);
	}

	/// @brief Run optimization test variant 3.
	///
	/// Assume only a single state exists.
	/// 1. Build test context.
	/// 2. Use vertex position as initial guess.
	/// 3. Set gradient as velocity.
	/// 4. Check convergence through verify_adjoint.
	///
	/// @param[in] json Filename of the input json config.
	/// @param[in] dt Time step.
	/// @param[in] tol Convergence check tolerance.
	void run_test3(const std::string &json_name, float dt, float tol)
	{
		TestContext ctx{json_name};

		// Compute initial solution.
		Eigen::MatrixXd V;
		ctx.states[0]->get_vertices(V);
		Eigen::VectorXd x = utils::flatten(V);

		ctx.problem->solution_changed(x);
		Eigen::VectorXd one_form;
		ctx.problem->gradient(x, one_form);

		verify_adjoint(*ctx.problem, x, one_form.normalized(), dt, tol);
	}

} // namespace

TEST_CASE("laplacian", "[test_adjoint]")
{
	TestContext ctx{"laplacian-opt.json"};

	Eigen::VectorXd x = ctx.var2sim.data[0]->inverse_eval();
	ctx.var2sim.update(x);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = vel(i) * std::cos(vel(i));
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	sample_field(*(ctx.states[0]), velocity, velocity_discrete);

	verify_adjoint(*ctx.problem, x, velocity_discrete, 1e-7, 1e-8);
}

TEST_CASE("linear_elasticity-surface-3d", "[test_adjoint]")
{
	run_test2("linear_elasticity-surface-3d-opt.json", 1e-7, 1e-5, -1.0, 1.0);
}

TEST_CASE("linear_elasticity-surface", "[test_adjoint]")
{
	run_test2("linear_elasticity-surface-opt.json", 1e-6, 1e-5, -1.0, 1.0);
}

TEST_CASE("topology-compliance", "[test_adjoint]")
{
	run_test1("topology-compliance-opt.json", 1e-2, 1e-4, 0.0, 1.0);
}

#if defined(NDEBUG) && !defined(WIN32)
std::string tagsdiff = "[test_adjoint]";
#else
std::string tagsdiff = "[.][test_adjoint]";
#endif

TEST_CASE("neohookean-stress-3d", tagsdiff)
{
	run_test2("neohookean-stress-3d-opt.json", 1e-7, 1e-4, -1.0, 1.0);
}

TEST_CASE("shape-neumann-nodes", "[test_adjoint]")
{
	run_test1("shape-neumann-nodes-opt.json", 1e-7, 1e-2, 0.0, 1.0);
}

TEST_CASE("shape-pressure-nodes-2d", "[test_adjoint]")
{
	run_test1("shape-pressure-nodes-2d-opt.json", 1e-7, 1e-2, 0.0, 1.0);
}

TEST_CASE("static-control-pressure-nodes-3d", "[.][test_adjoint]")
{
	run_test2("static-control-pressure-nodes-3d-opt.json", 1e-3, 1e-3, 0.0, 1.0);
}

TEST_CASE("control-pressure-walker-2d", "[test_adjoint]")
{
	run_test1("walker-opt.json", 1e-4, 1e-3, 0.0, 1.0);
}

TEST_CASE("shape-walker-2d", "[test_adjoint]")
{
	run_test1("walker-shape-opt.json", 1e-7, 1e-3, 0.0, 1.0);
}

TEST_CASE("shape-contact-force-norm", "[test_adjoint]")
{
	run_test1("shape-contact-force-norm-opt.json", 1e-7, 1e-3, 0.0, 1.0);
}

TEST_CASE("shape-contact-force-norm-adhesion", "[test_adjoint]")
{
	run_test1("shape-contact-force-norm-opt-adhesion.json", 1e-7, 1e-1, 0.0, 1.0);
}

TEST_CASE("shape-contact-force-norm-3d", "[test_adjoint]")
{
	run_test1("shape-contact-force-norm-3d-opt.json", 1e-6, 1e-3, 0.0, 1.0);
}

TEST_CASE("shape-contact", "[test_adjoint]")
{
	run_test3("shape-contact-opt.json", 1e-7, 1e-5);
}

TEST_CASE("shape-contact-adhesion", "[test_adjoint]")
{
	run_test3("shape-contact-adhesion-opt.json", 1e-7, 1e-5);
}

TEST_CASE("node-trajectory", "[test_adjoint]")
{
	// Prepare test manually because we need random target form.
	// The opt json is mostly a dummy json.

	json opt_args;
	load_json(append_root_path("node-trajectory-opt.json"), opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	// One state only.
	std::string root = POLYFEM_DIFF_DIR + std::string("/input/");
	auto states =
		from_json::build_states(root, opt_args["states"], -1);
	std::vector<std::shared_ptr<DiffCache>> diff_caches = {std::make_shared<DiffCache>()};

	auto elastic_var2sim =
		std::make_shared<ElasticVariableToSimulation>(states, diff_caches, CompositeParametrization{});
	VariableToSimulationGroup var2sim_group;
	var2sim_group.data.push_back(elastic_var2sim);

	std::srand(Catch::rngSeed());
	Eigen::MatrixXd targets =
		Eigen::MatrixXd::Random(states[0]->n_bases, states[0]->mesh->dimension());
	targets = (targets.array() + 1.0f) * 0.5f * 10.0f;

	// All actice.
	std::vector<int> actives(targets.rows());
	for (int i = 0; i < targets.rows(); ++i)
	{
		actives[i] = i;
	}

	auto form =
		std::make_shared<NodeTargetForm>(states[0], diff_caches[0], var2sim_group, actives, targets);
	AdjointNLProblem problem{form, var2sim_group, states, diff_caches, opt_args};
	Eigen::VectorXd x = var2sim_group.data[0]->inverse_eval();
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Random(x.size(), 1);

	verify_adjoint(problem, x, velocity, 1e-5, 1e-4);
}

TEST_CASE("damping-transient", "[test_adjoint]")
{
	run_test1("damping-transient-opt.json", 1e-3, 1e-4, 1.0, 1.0);
}

TEST_CASE("material-transient", "[test_adjoint]")
{
	run_test1("material-transient-opt.json", 1e-5, 1e-4, 1e3, 1e3);
}

TEST_CASE("shape-transient-friction", "[test_adjoint]")
{
	run_test2("shape-transient-friction-opt.json", 1e-6, 1e-5, 0.0, 1.0);
}

TEST_CASE("shape-transient-friction-sdf", "[test_adjoint]")
{
	run_test2("shape-transient-friction-sdf-opt.json", 1e-7, 1e-5, 0.0, 1.0);
}

TEST_CASE("3d-shape-mesh-target", "[.][test_adjoint]")
{
	run_test1("3d-shape-mesh-target-opt.json", 1e-7, 1e-5, 0.0, 1.0);
}

TEST_CASE("initial-contact-min-dist", "[test_adjoint]")
{
	run_test1("initial-contact-min-dist-opt.json", 1e-5, 1e-5, -1.0, 1.0);
}

TEST_CASE("friction-contact", "[test_adjoint]")
{
	TestContext ctx{"friction-contact-opt.json"};

	Eigen::VectorXd velocity = Eigen::VectorXd::Ones(1);
	Eigen::VectorXd x = 0.2f * Eigen::VectorXd::Ones(1);

	verify_adjoint(*ctx.problem, x, velocity, 1e-4, 1e-6);
}

TEST_CASE("barycenter", "[test_adjoint]")
{
	TestContext ctx{"barycenter-opt.json"};

	Eigen::MatrixXd velocity = Eigen::MatrixXd::Zero(ctx.states[0]->ndof() * 2, 1);
	for (int i = 0; i < ctx.states[0]->n_bases; i++)
	{
		velocity(ctx.states[0]->ndof() + i * 2 + 0) = -2.0f;
		velocity(ctx.states[0]->ndof() + i * 2 + 1) = -1.0f;
	}

	Eigen::VectorXd x = ctx.var2sim.data[0]->inverse_eval();

	verify_adjoint(*ctx.problem, x, velocity, 1e-6, 1e-5);
}

TEST_CASE("shape-contact-smooth", "[test_adjoint]")
{
	TestContext ctx{"shape-contact-opt.json"};

	// Because state configs are shared, tailor json args.
	for (auto &state : ctx.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["use_convergent_formulation"] = false;
		state->args["contact"]["alpha_t"] = 0.95;
	}

	Eigen::MatrixXd V;
	ctx.states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	ctx.problem->solution_changed(x);
	Eigen::VectorXd one_form;
	ctx.problem->gradient(x, one_form);

	verify_adjoint(*ctx.problem, x, one_form.normalized(), 1e-6, 1e-5);
}

TEST_CASE("initial-contact-smooth", "[test_adjoint]")
{
	TestContext ctx{"initial-contact-smooth-opt.json"};

	// Because state configs are shared, tailor json args.
	for (auto &state : ctx.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["use_convergent_formulation"] = false;
		state->args["contact"]["alpha_t"] = 0.95;
		state->args["contact"]["friction_coefficient"] = 0;
	}

	std::srand(Catch::rngSeed());
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Random(ctx.ndof, 1);
	Eigen::VectorXd x = ctx.var2sim.data[0]->inverse_eval();

	verify_adjoint(*ctx.problem, x, velocity, 1e-6, 1e-4);
}

TEST_CASE("shape-transient-smooth", "[test_adjoint]")
{
	TestContext ctx{"shape-transient-friction-opt.json"};

	// Because states are shared, tailor json args.
	for (auto &state : ctx.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["alpha_t"] = 0.95;
		state->args["contact"]["friction_coefficient"] = 0;
		state->args["solver"]["nonlinear"]["grad_norm"] = 1e-6;
	}

	std::srand(Catch::rngSeed());
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Random(ctx.ndof, 1);
	velocity.normalize();

	Eigen::MatrixXd V;
	ctx.states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	verify_adjoint(*ctx.problem, x, velocity, 1e-6, 1e-5);
}

TEST_CASE("shape-pressure-nodes-3d", "[.][test_adjoint]")
{
	run_test1("shape-pressure-nodes-3d-opt.json", 1e-7, 1e-3, 0.0, 1.0);
}

TEST_CASE("control-pressure-nodes-3d", "[.][test_adjoint]")
{
	run_test1("control-pressure-nodes-3d-opt.json", 1e-8, 1e-3, 0.0, 1.0);
}

TEST_CASE("dirichlet-nodes-3d", "[.][test_adjoint]")
{
	TestContext ctx{"dirichlet-nodes-3d-opt.json"};

	Eigen::VectorXd x(12);
	// clang-format off
	x << 0.0f,  0.0f, 0.0f,
	     0.0f,  0.0f, 0.0f,
	     0.0f, -0.2f, 0.0f,
	     0.0f, -0.2f, 0.0f;
	// clang-format on

	std::srand(Catch::rngSeed());
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Random(x.size(), 1);

	verify_adjoint(*ctx.problem, x, velocity, 1e-7, 1e-3);
}
