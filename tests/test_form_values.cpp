////////////////////////////////////////////////////////////////////////////////
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>

#include <polyfem/time_integrator/ImplicitEuler.hpp>

#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <catch2/catch.hpp>
#include <iostream>
#include <memory>
#include <filesystem>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::solver;
using namespace polyfem::time_integrator;
using namespace polyfem::assembler;

// map BroadPhaseMethod values to JSON as strings
namespace ipc
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "sweep_and_tiniest_queue_gpu"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}})
} // namespace ipc

namespace
{
	std::shared_ptr<State> get_state(const std::string &json_file)
	{
		const std::string path = POLYFEM_DATA_DIR;
		json in_args = json({});

		std::ifstream file(path + "/" + json_file);

		if (file.is_open())
			file >> in_args;

		in_args["root_path"] = path + "/" + json_file;

		auto state = std::make_shared<State>(1);
		state->init_logger("", spdlog::level::warn, false);
		state->init(in_args, true, "", true);

		state->load_mesh();

		state->build_basis();
		state->assemble_rhs();
		state->assemble_stiffness_mat();

		return state;
	}

	void check_form_value(
		const std::function<std::shared_ptr<Form>(const std::shared_ptr<const State>)> &create_form,
		const std::string &expected_key)
	{
		const std::string path = POLYFEM_DATA_DIR;
		const std::string pattern = path + "/forms/";

		for (auto const &dir_entry : std::filesystem::directory_iterator{pattern})
		{
			if (dir_entry.path().extension() != ".hdf5")
				continue;

			std::cout << "Processing: " << dir_entry.path() << std::endl;
			HighFive::File file(dir_entry.path(), HighFive::File::ReadOnly);
			std::string json_path = H5Easy::load<std::string>(file, "json_path");

			const std::vector<std::string> call_stack = H5Easy::load<std::vector<std::string>>(file, "call_stack");

			Eigen::MatrixXd val;

			const std::shared_ptr<const State> state = get_state(json_path);
			std::shared_ptr<Form> form = create_form(state);

			for (int i = 0; i < call_stack.size(); ++i)
			{
				const auto &call = call_stack[i];

				if (call.rfind("init_", 0) == 0)
				{
					const Eigen::MatrixXd x = H5Easy::load<Eigen::MatrixXd>(file, call);
					form->init(x);
				}
				else if (call.rfind("set_project_to_psd_", 0) == 0)
				{
					const bool val = H5Easy::load<bool>(file, call);
					form->set_project_to_psd(val);
				}
				else if (call.rfind("init_lagging_", 0) == 0)
				{
					const Eigen::MatrixXd x = H5Easy::load<Eigen::MatrixXd>(file, call);
					form->init_lagging(x);
				}
				else if (call.rfind("update_quantities_", 0) == 0)
				{
					const Eigen::MatrixXd x = H5Easy::load<Eigen::MatrixXd>(file, call);
					REQUIRE(call_stack[i + 1].rfind("update_quantities_t_", 0) == 0);
					const double t = H5Easy::load<double>(file, call_stack[i + 1]);
					form->update_quantities(t, x);
					++i;
				}
				else if (call.rfind("line_search_begin_0_", 0) == 0)
				{
					const Eigen::MatrixXd x0 = H5Easy::load<Eigen::MatrixXd>(file, call);
					REQUIRE(call_stack[i + 1].rfind("line_search_begin_1_", 0) == 0);
					const Eigen::MatrixXd x1 = H5Easy::load<Eigen::MatrixXd>(file, call_stack[i + 1]);
					form->line_search_begin(x0, x1);
					++i;
				}
				else if (call.rfind("line_search_end_", 0) == 0)
				{
					form->line_search_end();
				}
				else if (call.rfind("max_step_size_0_", 0) == 0)
				{
					const Eigen::MatrixXd x0 = H5Easy::load<Eigen::MatrixXd>(file, call);
					REQUIRE(call_stack[i + 1].rfind("max_step_size_1_", 0) == 0);
					const Eigen::MatrixXd x1 = H5Easy::load<Eigen::MatrixXd>(file, call_stack[i + 1]);
					form->max_step_size(x0, x1);
					++i;
				}
				else if (call.rfind("is_step_valid_0_", 0) == 0)
				{
					const Eigen::MatrixXd x0 = H5Easy::load<Eigen::MatrixXd>(file, call);
					REQUIRE(call_stack[i + 1].rfind("is_step_valid_1_", 0) == 0);
					const Eigen::MatrixXd x1 = H5Easy::load<Eigen::MatrixXd>(file, call_stack[i + 1]);
					form->is_step_valid(x0, x1);
					++i;
				}
				else if (call.rfind("value_", 0) == 0)
				{
					val = H5Easy::load<Eigen::MatrixXd>(file, call);
				}
				else if (call.rfind("solution_changed_", 0) == 0)
				{
					const Eigen::MatrixXd x = H5Easy::load<Eigen::MatrixXd>(file, call);
					form->solution_changed(x);
				}
				else if (call.rfind("post_step_", 0) == 0)
				{
					const Eigen::MatrixXd x = H5Easy::load<Eigen::MatrixXd>(file, call);
					REQUIRE(call_stack[i + 1].rfind("post_step_iter_", 0) == 0);
					const int iter = H5Easy::load<int>(file, call_stack[i + 1]);
					form->post_step(iter, x);
					++i;
				}
				else if (call.rfind(expected_key, 0) == 0)
				{
					const double expected = H5Easy::load<double>(file, call);
					REQUIRE(val.size() > 0);
					const double value = form->value(val);
					if (!std::isnan(expected))
						REQUIRE(value == Approx(expected).epsilon(1e-6).margin(1e-9));
					else
						REQUIRE(std::isnan(value));

					val.resize(0, 0);
				}
				else
				{
					static const std::vector<std::string> tmp = {"elastic_energy_", "body_energy_", "friction_energy_", "lagged_damping_", "intertia_energy_", "barrier_stiffness_", "collision_energy_"};
					bool found = false;

					for (const auto &v : tmp)
					{
						if (call.rfind(v, 0) == 0)
						{
							found = true;
							break;
						}
					}

					if (!found)
						std::cerr << call << " not found" << std::endl;
					REQUIRE(found);
				}
			}
		}
	}
} // namespace

TEST_CASE("body form value", "[form][form_value][body_form]")
{
	std::shared_ptr<assembler::RhsAssembler> rhs_assembler;
	const bool apply_DBC = true;

	CAPTURE(apply_DBC);
	check_form_value(
		[&](const std::shared_ptr<const State> state) {
			rhs_assembler = state->build_rhs_assembler();
			return std::make_shared<BodyForm>(*state, *rhs_assembler, apply_DBC);
		},
		"body_energy");
}

TEST_CASE("contact form value", "[form][form_value][contact_form]")
{
	std::shared_ptr<assembler::RhsAssembler> rhs_assembler;
	const bool apply_DBC = true;
	std::shared_ptr<BodyForm> body_form;

	const double barrier_stiffness = 1e7;
	// const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;

	check_form_value(
		[&](const std::shared_ptr<const State> state) {
			rhs_assembler = state->build_rhs_assembler();
			body_form = std::make_shared<BodyForm>(*state, *rhs_assembler, apply_DBC);
			const double dt = state->args["time"]["dt"];
			return std::make_shared<ContactForm>(
				*state,
				state->args["contact"]["dhat"],
				// f.create_dataset("barrier_stiffness", data = barrier_stiffness)
				!state->args["solver"]["contact"]["barrier_stiffness"].is_number(),
				state->problem->is_time_dependent(),
				state->args["solver"]["contact"]["CCD"]["broad_phase"],
				state->args["solver"]["contact"]["CCD"]["tolerance"],
				state->args["solver"]["contact"]["CCD"]["max_iterations"],
				dt * dt,
				*body_form);
		},
		"collision_energy");
}

TEST_CASE("elastic form value", "[form][form_value][elastic_form]")
{
	check_form_value(
		[](const std::shared_ptr<const State> state) {
			return std::make_shared<ElasticForm>(*state);
		},
		"elastic_energy");
}

TEST_CASE("friction form value", "[form][form_value][friction_form]")
{
	std::shared_ptr<assembler::RhsAssembler> rhs_assembler;
	const bool apply_DBC = true;
	std::shared_ptr<BodyForm> body_form;

	const double barrier_stiffness = 1e7;
	std::shared_ptr<ContactForm> contact_form;

	check_form_value(
		[&](const std::shared_ptr<const State> state) {
			rhs_assembler = state->build_rhs_assembler();
			body_form = std::make_shared<BodyForm>(*state, *rhs_assembler, apply_DBC);

			const double dt = state->args["time"]["dt"];
			contact_form = std::make_shared<ContactForm>(
				*state,
				state->args["contact"]["dhat"],
				!state->args["solver"]["contact"]["barrier_stiffness"].is_number(),
				state->problem->is_time_dependent(),
				state->args["solver"]["contact"]["CCD"]["broad_phase"],
				state->args["solver"]["contact"]["CCD"]["tolerance"],
				state->args["solver"]["contact"]["CCD"]["max_iterations"],
				dt * dt,
				*body_form);

			return std::make_shared<FrictionForm>(
				*state,
				state->args["contact"]["epsv"],
				state->args["contact"]["friction_coefficient"],
				state->args["contact"]["dhat"],
				state->args["solver"]["contact"]["CCD"]["broad_phase"],
				dt,
				*contact_form);
		},
		"friction_energy");
}

TEST_CASE("inertia form value", "[form][form_value][inertia_form]")
{
	ImplicitEuler time_integrator;

	check_form_value(
		[&](const std::shared_ptr<const State> state) {
			const double dt = state->args["time"]["dt"];
			time_integrator.init(
				Eigen::VectorXd::Zero(state->n_bases * state->mesh->dimension()),
				Eigen::VectorXd::Zero(state->n_bases * state->mesh->dimension()),
				Eigen::VectorXd::Zero(state->n_bases * state->mesh->dimension()),
				dt);
			return std::make_shared<InertiaForm>(state->mass, time_integrator);
		},
		"intertia_energy");
}

TEST_CASE("lagged regularization form value", "[form][form_value][lagged_reg_form]")
{
	const double weight = 0.0;
	check_form_value(
		[&](const std::shared_ptr<const State>) {
			return std::make_shared<LaggedRegForm>(weight);
		},
		"lagged_damping");
}