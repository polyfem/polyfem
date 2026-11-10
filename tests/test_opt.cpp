////////////////////////////////////////////////////////////////////////////////
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/optimization/Optimizations.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <polyfem/optimization/BuildFromJson.hpp>
#include <polyfem/optimization/NodeSelectionUtils.hpp>
#include <polyfem/optimization/var2sims/ShapeVariableToSimulation.hpp>

#include <polyfem/optimization/forms/SumCompositeForm.hpp>
#include <polyfem/optimization/forms/CompositeForms.hpp>
#include <polyfem/optimization/forms/TargetForms.hpp>
#include <polyfem/optimization/forms/AMIPSForm.hpp>
#include <polyfem/optimization/forms/WeightedVolumeForm.hpp>
#include <polyfem/optimization/forms/TransientForm.hpp>
#include <polyfem/optimization/forms/BarrierForms.hpp>
#include <polyfem/optimization/forms/SmoothingForms.hpp>

#include <polyfem/optimization/parametrization/Parametrizations.hpp>
#include <polyfem/optimization/OptState.hpp>
#include <polyfem/io/InputLoader.hpp>

#include <polysolve/nonlinear/BoxConstraintSolver.hpp>

#include <fstream>
#include <functional>
#include <catch2/catch_all.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace solver;
using namespace polysolve;

// Some tests is computationally expensive.
// To avoid destroying CI time, we only test those in release build.
#ifdef NDEBUG
#define EXPENSIVE_TEST_LABEL "[optimization]"
#else
#define EXPENSIVE_TEST_LABEL "[.][optimization]"
#endif

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

	class TestContext
	{
	public:
		json args;
		OptState opt;
		std::vector<io::LoadedInput> loaded_states;

		explicit TestContext(const std::string &name)
		{
			const std::string root_folder = POLYFEM_DIFF_DIR + std::string("/optimizations/") + name + "/";
			if (!load_json(root_folder + "run.json", args))
			{
				log_and_throw_adjoint_error("Failed to load optimization json file!");
			}

			// Silence output.
			args["output"]["directory"] = "";
			args["output"]["save_frequency"] = 100000;
			args["output"]["log"]["path"] = "";
			args["output"]["log"]["level"] = "off";
			args["output"]["log"]["file_level"] = "off";
			args["output"]["log"]["quiet"] = true;
			args["solver"]["max_threads"] = 1;
			args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";

			opt.init(args, false);
			create_varforms_with_patch([](json &varform_args) {
				varform_args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
			});
			opt.init_variables();
			opt.create_problem();
		}

		std::shared_ptr<polysolve::nonlinear::Solver> make_solver() const
		{
			return AdjointOptUtils::make_nl_solver(
				opt.args["solver"]["nonlinear"],
				opt.args["solver"]["linear"],
				1);
		}

	private:
		void create_varforms_with_patch(const std::function<void(json &)> &patch)
		{
			const int max_threads = 1;
			opt.varforms.resize(opt.args["states"].size());
			opt.state_resources.resize(opt.args["states"].size());

			for (int i = 0; i < opt.args["states"].size(); ++i)
			{
				const std::string abs_path = utils::resolve_path(opt.args["states"][i]["path"], opt.args["root_path"], false);
				loaded_states.push_back(io::load_json_input(abs_path));
				json varform_args = loaded_states.back().config;

				if (!opt.args["output"]["log"].empty())
					varform_args["output"]["log"].merge_patch(opt.args["output"]["log"]);

				patch(varform_args);
				opt.state_resources[i] = loaded_states.back().resources->with_root("");
				opt.varforms[i] = from_json::build_differentiable_varform(
					varform_args, *opt.state_resources[i], max_threads);
			}

			opt.diff_caches.resize(opt.varforms.size());
			for (auto &diff_cache : opt.diff_caches)
				diff_cache = std::make_shared<DiffCache>();
		}
	};

	/// @brief Run test variant 1.
	/// @param name Test name.
	/// @param expected_energy Expected objective energy.
	/// @param epsilon Energy check tolerance rel tolerance.
	void run_test1(const std::string &name, const double expected_energy, double tol)
	{
		TestContext ctx{name};

		Eigen::VectorXd x;
		ctx.opt.initial_guess(x);

		auto nl_solver = ctx.make_solver();
		try
		{
			nl_solver->minimize(*ctx.opt.nl_problem, x);
		}
		catch (const std::runtime_error &e)
		{
			// Out if iteration error is fine as long as the expected energy match later.
			REQUIRE_THAT(
				std::string(e.what()),
				Catch::Matchers::ContainsSubstring("Reached iteration limit"));
		}

		const json &params = nl_solver->info();
		logger().trace("final energy {}", params["energy"].get<double>());

		REQUIRE(params["energy"].get<double>() == Catch::Approx(expected_energy).epsilon(tol));
	}

} // namespace

TEST_CASE("material-opt", EXPENSIVE_TEST_LABEL)
{
	run_test1("material-opt", 0.0023793444, 1e-2);
}

TEST_CASE("friction-opt", EXPENSIVE_TEST_LABEL)
{
	run_test1("friction-opt", 1.7981112e-09, 1e-1);
}

TEST_CASE("damping-opt", EXPENSIVE_TEST_LABEL)
{
	run_test1("damping-opt", 2.9914589897e-07, 1e-3);
}

TEST_CASE("initial-opt", "[optimization]")
{
	run_test1("initial-condition-trajectory-opt", 4.58399e-05, 1e-2);
}

TEST_CASE("topology-opt", "[optimization]")
{
	TestContext ctx{"topology-opt"};

	Eigen::VectorXd x;
	ctx.opt.initial_guess(x);

	auto nl_solver = ctx.make_solver();
	auto box_solver = std::dynamic_pointer_cast<polysolve::nonlinear::BoxConstraintSolver>(nl_solver);
	REQUIRE(box_solver != nullptr);

	// nonlinear inequality constraints g(x) < 0
	{
		auto obj1 = std::make_shared<WeightedVolumeForm>(
			CompositeParametrization({std::make_shared<LinearFilter>(ctx.opt.varforms[0]->get_mesh(), 0.1)}),
			ctx.opt.varforms[0]);
		obj1->set_weight(1 / 1.2);
		auto obj2 = std::make_shared<PlusConstCompositeForm>(obj1, -1);
		std::vector<std::shared_ptr<Form>> constraints = {{obj2}};
		auto constraint = std::make_shared<FullNLProblem>(constraints);
		box_solver->add_constraint(constraint);
	}

	try
	{
		nl_solver->minimize(*ctx.opt.nl_problem, x);
	}
	catch (const std::runtime_error &e)
	{
		REQUIRE_THAT(
			std::string(e.what()),
			Catch::Matchers::ContainsSubstring("Reached iteration limit"));
	}

	const json &params = nl_solver->info();
	logger().trace("final energy {}", params["energy"].get<double>());

	REQUIRE(params["energy"].get<double>() == Catch::Approx(0.726565).epsilon(1e-4));
}

TEST_CASE("AMIPS-debug", "[optimization]")
{
	run_test1("AMIPS-debug", 1.00006, 1e-4);
}

TEST_CASE("shape-stress-opt", EXPENSIVE_TEST_LABEL)
{
	run_test1("shape-stress-opt", 0.0587843446, 1e-4);
}

TEST_CASE("3d-shape-layer-thickness", EXPENSIVE_TEST_LABEL)
{
	run_test1("3d-shape-layer-thickness", 0.4881e-3, 8e-3);
}
