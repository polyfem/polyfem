////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include <polyfem/MaybeParallelFor.hpp>

#include <catch2/catch.hpp>

#include <math.h>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

TEST_CASE("ncmesh2d", "[ncmesh]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"problem": "GenericScalar",
			"scalar_formulation": "Laplacian",
            "export":{
                "high_order_mesh": false
            },
			"n_refs": 0,
			"discr_order": 2,
			"iso_parametric": false,
            "bc_method": "sample",
			"problem_params": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": "x^2+y^2"
				}],
                "exact": "x^2+y^2",
                "exact_grad": ["2*x","2*y"],
				"rhs": 4
			},
			"vismesh_rel_area": 1.0
		}
	)"_json;
	in_args["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";

	State state(8);
	state.init_logger("", 6, false);
	state.init(in_args);

	state.load_mesh(true);
	NCMesh2D &ncmesh = *dynamic_cast<NCMesh2D *>(state.mesh.get());

	for (int n = 0; n < 1; n++)
	{
		ncmesh.prepare_mesh();
		std::vector<int> ref_ids(ncmesh.n_faces() / 2);
		for (int i = 0; i < ref_ids.size(); i++)
			ref_ids[i] = 2 * i;

		ncmesh.refine_elements(ref_ids);
	}
	ncmesh.prepare_mesh();

	state.compute_mesh_stats();
	state.build_basis();

	state.assemble_stiffness_mat();
	state.assemble_rhs();

	state.solve_problem();
	state.compute_errors();

	REQUIRE(fabs(state.h1_semi_err) < 1e-9);
	REQUIRE(fabs(state.l2_err) < 1e-10);
}

TEST_CASE("ncmesh3d", "[ncmesh]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"problem": "GenericScalar",
			"scalar_formulation": "Laplacian",
            "export":{
                "high_order_mesh": false
            },
			"n_refs": 0,
			"solver_type": "Eigen::SimplicialLDLT",
			"discr_order": 2,
			"iso_parametric": true,
            "bc_method": "sample",
			"vismesh_rel_area": 1e-6,
			"problem_params": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": "x^2+y^2+z^2"
				}],
                "exact": "x^2+y^2+z^2",
                "exact_grad": ["2*x","2*y","2*z"],
				"rhs": 6
			}
		}
	)"_json;
	in_args["mesh"] = path + "/../../lbar-extrude.msh";

	State state(8);
	state.init_logger("", 1, false);
	state.init(in_args);

	state.load_mesh(true);
	NCMesh3D &ncmesh = *dynamic_cast<NCMesh3D *>(state.mesh.get());
	std::vector<int> parent_nodes;
	// ncmesh.refine(1, 0., parent_nodes);
	for (int n = 0; n < 2; n++)
	{
		ncmesh.prepare_mesh();
		std::vector<int> ref_ids(int(ncmesh.n_cells()/2.01));
		for (int i = 0; i < ref_ids.size(); i++)
			ref_ids[i] = i * 2;

		ncmesh.refine_elements(ref_ids);
	}
	ncmesh.prepare_mesh();

	state.compute_mesh_stats();
	state.build_basis();

	state.assemble_stiffness_mat();
	state.assemble_rhs();

	state.solve_problem();
	state.compute_errors();

	state.save_vtu(state.resolve_output_path("debug.vtu"), 1.);

	REQUIRE(fabs(state.h1_semi_err) < 1e-9);
	REQUIRE(fabs(state.l2_err) < 1e-10);
}
