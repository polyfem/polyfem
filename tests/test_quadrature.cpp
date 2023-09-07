////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/quadrature/LineQuadrature.hpp>
#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::quadrature;

const double pi = 3.14159265358979323846264338327950288419717;

////////////////////////////////////////////////////////////////////////////////

namespace
{

	// double p01_exact() {
	// 	return pi * pi / 6.0;
	// }

	// Eigen::VectorXd p01_fun(const Eigen::MatrixXd &x) {
	// 	Eigen::VectorXd f = 1.0 / (1.0 - x.row(0).array() * x.row(1).array());
	// 	return f;
	// }

	// Eigen::AlignedBox2d p01_lim() {
	// 	Eigen::AlignedBox2d box;
	// 	box.min().setConstant(0.0);
	// 	box.max().setConstant(1.0);
	// 	return box;
	// }

	namespace
	{
		std::shared_ptr<State> get_state(
			const std::string &mesh, const int n_refs,
			const int basis_order,
			const int quadrature, const int mass_quadrature,
			const bool spline, const bool serendipity)
		{
			const std::string path = POLYFEM_DATA_DIR;
			json in_args = R"(
		{
			"materials": {
                "type": "Laplacian"
            },

			"geometry": [{
				"mesh": "",
				"enabled": true,
				"type": "mesh",
				"surface_selection": 7,
				"n_refs": 0
			}],

			"space": {
				"discr_order": 0,
				"advanced": {
					"quadrature_order": -1,
					"mass_quadrature_order": -1
				}
			},

			"time": {
				"dt": 0.001,
				"tend": 1.0
			},

			"boundary_conditions": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": 0
				}],
				"rhs": 10
			},

			"output": {
				"log": {
					"level": "warning"
				}
			}

		})"_json;
			in_args["geometry"][0]["mesh"] = path + "/quad_test/" + mesh;
			in_args["geometry"][0]["n_refs"] = n_refs;
			in_args["space"]["discr_order"] = basis_order;
			in_args["space"]["advanced"]["quadrature_order"] = quadrature;
			in_args["space"]["advanced"]["mass_quadrature_order"] = mass_quadrature;

			if (spline)
				in_args["space"]["basis_type"] = "Spline";
			else if (serendipity)
				in_args["space"]["basis_type"] = "Serendipity";

			auto state = std::make_shared<State>();
			state->set_max_threads(1);
			state->init(in_args, true);

			state->load_mesh();

			state->build_basis();
			state->assemble_rhs();
			state->assemble_mass_mat();

			return state;
		}

		void test_quadrature(const std::string &mesh, const int n_refs,
							 const int basis_order,
							 const bool spline, const bool serendipity)
		{
			const bool is_q = mesh.find("quad") != std::string::npos || mesh.find("hex") != std::string::npos;
			const int expected_quad = is_q ? 20 : 14;
			static const double margin = 1e-10;
			// spdlog::info("Reference quad={}", expected_quad);

			auto state = get_state(mesh, n_refs, basis_order, -1, -1, spline, serendipity);
			auto expected = get_state(mesh, n_refs, basis_order, expected_quad, expected_quad, spline, serendipity);
			StiffnessMatrix exp_st, st;
			state->build_stiffness_mat(st);
			expected->build_stiffness_mat(exp_st);

			StiffnessMatrix tmp = st - exp_st;
			const auto val = Catch::Approx(0).margin(margin);

			REQUIRE(tmp.rows() > 8);

			for (int k = 0; k < tmp.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(tmp, k); it; ++it)
				{
					if (fabs(it.value()) > margin)
						spdlog::error("error: {} != 0", it.value());
					REQUIRE(it.value() == val);
				}
			}

			tmp = state->mass - expected->mass;

			for (int k = 0; k < tmp.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(tmp, k); it; ++it)
				{
					if (fabs(it.value()) > margin)
						spdlog::error("error: {} != 0", it.value());
					REQUIRE(it.value() == val);
				}
			}
		}
	} // namespace

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

TEST_CASE("auto_quadrature", "[quadrature]")
{
	struct data
	{
		std::string mesh;
		int n_refs;
		int order;
		bool spline;
		bool serendipity;
	};

	std::vector<data> tests = {
		// P
		{"tri.obj", 2, 1, false, false},
		{"tri.obj", 1, 2, false, false},
		{"tri.obj", 1, 3, false, false},

		{"tet.msh", 0, 1, false, false},
		{"tet.msh", 0, 2, false, false},

		// Q
		{"quad.obj", 3, 1, false, false},
		{"quad.obj", 2, 2, false, false},

		{"hex.HYBRID", 0, 1, false, false},
		{"hex.HYBRID", 0, 2, false, false},

		// Spline
		{"quad.obj", 2, 2, true, false},
		{"hex.HYBRID", 0, 2, true, false},

		// serendipity
		{"quad.obj", 2, 2, false, true},
		{"hex.HYBRID", 0, 2, false, true},
	};

	for (const auto &d : tests)
	{
		spdlog::set_level(spdlog::level::info);
		spdlog::info("Running {} Order={}, spline={} serendipity={}", d.mesh, d.order, d.spline, d.serendipity);
		test_quadrature(d.mesh, d.n_refs, d.order, d.spline, d.serendipity);
	}
}

TEST_CASE("weights", "[quadrature]")
{
	// Segment
	for (int order = 1; order <= 64; ++order)
	{
		LineQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(1.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Triangle
	for (int order = 1; order < 16; ++order)
	{
		TriQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(0.5).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Tetrahedron
	for (int order = 1; order < 16; ++order)
	{
		TetQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(1.0 / 6.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}
}

// TEST_CASE("triangle", "[quadrature]") {
//	for (int order = 1; order < 10; ++order) {
//		Quadrature quadr;
//		TriQuadrature tri;
//		tri.get_quadrature(order, quadr);
//	}
//
//	// REQUIRE(poly_fem::determinant(mat) == Catch::Approx(mat.determinant()).margin(1e-12));
// }
