////////////////////////////////////////////////////////////////////////////////
#include <polyfem/solver/forms/ElasticForm.hpp>

#include <catch2/catch.hpp>
#include <iostream>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::solver;

std::shared_ptr<State> get_state()
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"materials": {
                "type": "NeoHookean",
                "E": 20000,
                "nu": 0.3,
                "rho": 1000
            },

			"geometry": [{
				"mesh": "",
				"enabled": true,
				"type": "mesh",
				"surface_selection": 7
			}],

			"boundary_conditions": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": [0, 0]
				}],
				"rhs": [10, 10]
			}

		})"_json;
	in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";

	auto state = std::make_shared<State>(1);
	state->init_logger("", spdlog::level::debug, false);
	state->init(in_args, true);

	state->load_mesh();

	state->build_basis();
	state->assemble_rhs();
	state->assemble_stiffness_mat();

	return state;
}

template <typename Form>
void test_form(Form &form, const State &state)
{
	static const int n_rand = 10;
	static const double eps = 1e-7;
	static const double margin = 1e-2;

	Eigen::MatrixXd x(state.n_bases * 2, 1);
	x.setZero();

	Eigen::VectorXd grad;

	for (int rand = 0; rand < n_rand; ++rand)
	{
		const double energy = form.value(x);
		form.first_derivative(x, grad);

		for (int d = 0; d < x.size(); ++d)
		{
			Eigen::MatrixXd d_x = x;
			d_x(d) += eps;

			const double d_energy = form.value(d_x);
			const double fd = (d_energy - energy) / eps;
			if (rand == 0) // zero displacement is a minimum, grad should be zero
				REQUIRE(grad(d) == Approx(0).margin(1e-10));
			else
				REQUIRE(grad(d) == Approx(fd).margin(margin));
		}

		x.setRandom();
		x /= 100;
	}
}

TEST_CASE("elastic", "[form]")
{
	const auto state_ptr = get_state();
	ElasticForm form(*state_ptr);

	test_form(form, *state_ptr);
}
