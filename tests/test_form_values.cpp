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
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}});
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
			std::string json_path = H5Easy::load<std::string>(file, "path");

			Eigen::MatrixXd x = H5Easy::load<Eigen::MatrixXd>(file, "vals");
			Eigen::MatrixXd expected = H5Easy::load<Eigen::MatrixXd>(file, expected_key);

			const std::shared_ptr<const State> state = get_state(json_path);
			std::shared_ptr<Form> form = create_form(state);

			assert(x.rows() == expected.size());

			for (int i = 0; i < expected.size(); ++i)
			{
				form->init(x.row(i));
				form->init_lagging(Eigen::VectorXd::Zero(x.cols())); // TODO

				const double val = form->value(x.row(i));
				if (!std::isnan(val) && !std::isnan(expected(i)))
					CHECK(val == Approx(expected(i)).epsilon(1e-6).margin(1e-9));
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