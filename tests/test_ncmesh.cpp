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
			"discr_order": 5,
			"iso_parametric": false,
            "bc_method": "sample",
			"problem_params": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": "x^5+y^5"
				}],
                "exact": "x^5+y^5",
                "exact_grad": ["5*x^4","5*y^4"],
				"rhs": "20*x^3+20*y^3"
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
			"discr_order": 5,
			"iso_parametric": true,
            "bc_method": "sample",
			"vismesh_rel_area": 1,
			"problem_params": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": "x^5+y^5+z^5"
				}],
                "exact": "x^5+y^5+z^5",
                "exact_grad": ["5*x^4","5*y^4","5*z^4"],
				"rhs": "20*x^3+20*y^3+20*z^3"
			}
		}
	)"_json;
	in_args["mesh"] = path + "/contact/meshes/3D/simple/bar/bar-186.msh";

	State state(8);
	state.init_logger("", 6, false);
	state.init(in_args);

	state.load_mesh(true);
	NCMesh3D &ncmesh = *dynamic_cast<NCMesh3D *>(state.mesh.get());
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

	REQUIRE(fabs(state.h1_semi_err) < 1e-7);
	REQUIRE(fabs(state.l2_err) < 1e-10);
}
