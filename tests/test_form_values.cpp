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
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::solver;
using namespace polyfem::time_integrator;
using namespace polyfem::assembler;

namespace
{
	std::shared_ptr<State> get_state(const std::string &json_file)
	{
		const std::string path = POLYFEM_DATA_DIR;
		json in_args = json({});

		std::ifstream file(path + json_file);

		if (file.is_open())
			file >> in_args;

		in_args["root_path"] = path + json_file;

		auto state = std::make_shared<State>(1);
		state->init_logger("", spdlog::level::warn, false);
		state->init(in_args, true, "", true);

		state->load_mesh();

		state->build_basis();
		state->assemble_rhs();
		state->assemble_stiffness_mat();

		return state;
	}
} // namespace

TEST_CASE("body form value", "[form_value]")
{
	const std::string path = POLYFEM_DATA_DIR;
	const std::string pattern = path + "forms/";

	for (auto const &dir_entry : std::filesystem::directory_iterator{pattern})
	{
		if (dir_entry.path().extension() != ".hdf5")
			continue;

		std::cout << "Processing: " << dir_entry.path() << std::endl;
		HighFive::File file(dir_entry.path(), HighFive::File::ReadOnly);
		std::string json_path = H5Easy::load<std::string>(file, "path");

		auto state = get_state(json_path);

		Eigen::MatrixXd x = H5Easy::load<Eigen::MatrixXd>(file, "vals");
		Eigen::MatrixXd expected = H5Easy::load<Eigen::MatrixXd>(file, "elastic_energy");

		ElasticForm form(*state);

		assert(x.rows() == expected.size());

		for (int i = 0; i < expected.size(); ++i)
		{
			const double val = form.value(x.row(i));
			REQUIRE(val == Approx(expected(i)).margin(1e-9));
		}
	}
}

// f.create_dataset("vals", data = vals)
// f.create_dataset("elastic_energy", data = elastic_energy)
// f.create_dataset("body_energy", data = body_energy)
// f.create_dataset("friction_energy", data = friction_energy)
// f.create_dataset("lagged_damping", data = lagged_damping)
// f.create_dataset("intertia_energy", data = intertia_energy)
// f.create_dataset("barrier_stiffness", data = barrier_stiffness)
// f.create_dataset("collision_energy", data = collision_energy)
// f.create_dataset("path", data = path)