#include <polyfem/State.hpp>

#include <catch2/catch.hpp>
#include <iostream>

#include <finitediff.hpp>

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;
using namespace polyfem::utils;

TEST_CASE("hessian_lin", "[assembler]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = json({});
	in_args["geometry"] = {};
	in_args["geometry"]["mesh"] = path + "/plane_hole.obj";
	in_args["geometry"]["surface_selection"] = 7;
	// in_args["geometry"]["mesh"] = path + "/circle2.msh";
	// in_args["force_linear_geometry"] = true;

	in_args["preset_problem"] = {};
	in_args["preset_problem"]["type"] = "ElasticExact";

	in_args["materials"] = {};
	in_args["materials"]["type"] = "LinearElasticity";
	in_args["materials"]["E"] = 1e5;
	in_args["materials"]["nu"] = 0.3;

	State state;
	state.init_logger("", spdlog::level::err, false);
	state.init(in_args, true);
	state.load_mesh();

	// state.compute_mesh_stats();
	state.build_basis();

	state.assemble_stiffness_mat();

	SpareMatrixCache mat_cache;
	StiffnessMatrix hessian;
	Eigen::MatrixXd disp(state.n_bases * 2, 1);
	disp.setZero();

	for (int rand = 0; rand < 10; ++rand)
	{
		state.assembler.assemble_energy_hessian(
			"LinearElasticity", false, state.n_bases, false,
			state.bases, state.bases, state.ass_vals_cache, 0, disp, Eigen::MatrixXd(), mat_cache, hessian);

		const StiffnessMatrix tmp = state.stiffness - hessian;
		const auto val = Approx(0).margin(1e-8);

		for (int k = 0; k < tmp.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(tmp, k); it; ++it)
			{
				REQUIRE(it.value() == val);
			}
		}

		disp.setRandom();
	}
}

TEST_CASE("multiscale_derivatives", "[assembler]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"geometry": [
			{
				"mesh": "",
				"transformation": {
					"scale": 1
				},
				"volume_selection": 1,
				"surface_selection": [
					{
						"id": 2,
						"axis": "-y",
						"position": -0.99
					},
					{
						"id": 4,
						"axis": "y",
						"position": 0.99
					},
					{
						"id": 1,
						"axis": "-x",
						"position": -0.99
					},
					{
						"id": 3,
						"axis": "x",
						"position": 0.99
					}
				]
			}
		],
		"solver": {
			"linear": {
				"solver": "Eigen::SimplicialLDLT"
			}
		},
		"boundary_conditions": {
			"periodic_boundary": [true, true]
		},
		"materials": {
			"type": "MultiscaleRB",
			"microstructure": {},
			"rho": 1
		}
	}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "../square.msh";

	json tmp = in_args;
	tmp["materials"]["type"] = "NeoHookean";
	tmp["materials"]["E"] = 100;
	tmp["materials"]["nu"] = 0.4;
	tmp["materials"].erase("microstructure");
	tmp["geometry"][0]["mesh"] = path + "../negative-nu.msh";
	tmp["geometry"][0]["transformation"]["scale"] = 0.01;
	in_args["materials"]["microstructure"] = tmp;

	State state;
	state.init_logger("", spdlog::level::info, false);
	state.init(in_args, false);
	state.load_mesh();
	state.build_basis();

	Eigen::MatrixXd grad;
	Eigen::MatrixXd disp(state.n_bases * 2, 1);
	disp.setZero();

	for (int rand = 0; rand < 5; ++rand)
	{
		state.assembler.assemble_energy_gradient(
			state.formulation(), false, state.n_bases, state.bases, state.geom_bases(), 
			state.ass_vals_cache, 0, disp, disp, grad);

		Eigen::VectorXd fgrad;
		fd::finite_gradient(
			disp, [&state](const Eigen::VectorXd &x) -> double { return state.assembler.assemble_energy(state.formulation(), false, state.bases, state.geom_bases(), state.ass_vals_cache, 0, x, x); }, fgrad);

		REQUIRE (fd::compare_gradient(grad, fgrad));

		disp.setRandom();
		disp /= 10;
	}
}