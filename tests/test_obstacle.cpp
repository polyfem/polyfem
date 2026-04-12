////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>

#include <polyfem/State.hpp>

////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

TEST_CASE("obstacle displacement is not injected into rhs through set_bc", "[obstacle]")
{
	json in_args = R"(
		{
			"materials": {
				"type": "LinearElasticity",
				"E": 1000,
				"nu": 0.3,
				"rho": 1000
			},
			"geometry": [{
				"mesh": "",
				"type": "mesh",
				"enabled": true
			}, {
				"mesh": "",
				"type": "mesh",
				"enabled": true,
				"is_obstacle": true,
				"surface_selection": 1,
				"transformation": {
					"translation": [0.0, 1.25]
				}
			}],
			"space": {
				"discr_order": 1,
				"advanced": {
					"bc_method": "sample"
				}
			},
			"boundary_conditions": {
				"rhs": [0, 0],
				"obstacle_displacements": [{
					"id": 1,
					"value": [0.25, -0.5]
				}]
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				}
			}
		})"_json;

	in_args["geometry"][0]["mesh"] =
		std::string(POLYFEM_DATA_DIR) + "/contact/meshes/2D/simple/square.obj";
	in_args["geometry"][1]["mesh"] =
		std::string(POLYFEM_DATA_DIR) + "/contact/meshes/2D/simple/square.obj";

	State state;
	state.set_max_threads(1);
	state.init_logger("", spdlog::level::off, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh(true);
	state.build_basis();
	state.assemble_rhs();

	REQUIRE(state.solve_data.rhs_assembler != nullptr);
	REQUIRE(state.obstacle.n_vertices() > 0);

	Eigen::MatrixXd rhs = state.rhs;
	const Eigen::MatrixXd rhs_before = rhs;

	state.solve_data.rhs_assembler->set_bc(
		state.local_boundary,
		state.boundary_nodes,
		state.n_boundary_samples(),
		state.local_neumann_boundary,
		rhs,
		Eigen::MatrixXd(),
		0.5);

	REQUIRE(rhs.rows() >= state.obstacle.ndof());
	const Eigen::MatrixXd obstacle_rows_after = rhs.bottomRows(state.obstacle.ndof());
	const Eigen::MatrixXd obstacle_rows_before = rhs_before.bottomRows(state.obstacle.ndof());
	REQUIRE((obstacle_rows_after - obstacle_rows_before).cwiseAbs().maxCoeff() == 0.0);
}
