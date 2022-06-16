////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include <polyfem/utils/MaybeParallelFor.hpp>

#include <catch2/catch.hpp>

#include <math.h>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;

TEST_CASE("ncmesh2d", "[ncmesh]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"materials": {"type": "Laplacian"},

			"geometry": [{
				"mesh": "",
				"enabled": true,
				"type": "mesh"
			}],

			"space":{
				"discr_order": 5,
				"advanced": {
					"isoparametric": false,
					"bc_method": "sample"
				}
			},

			"boundary_conditions": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": "x^5+y^5"
				}],
				"rhs": "20*x^3+20*y^3"
			},

			"output": {
				"reference": {
	            	"solution": "x^5+y^5",
	            	"gradient": ["5*x^4","5*y^4"]
				},
				"paraview": {
					"high_order_mesh": false
				}
			}
		})"_json;
	in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";

	State state(1);
	state.init_logger("", 6, false);
	state.init(in_args);

	state.load_mesh(true);
	NCMesh2D &ncmesh = *dynamic_cast<NCMesh2D *>(state.mesh.get());

	for (int n = 0; n < 2; n++)
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

	// state.save_vtu("debug.vtu", 1.);

	REQUIRE(fabs(state.h1_semi_err) < 1e-9);
	REQUIRE(fabs(state.l2_err) < 1e-10);
}

TEST_CASE("ncmesh3d", "[ncmesh]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"materials": {"type": "Laplacian"},

			"geometry": [{
				"mesh": "",
				"enabled": true,
				"type": "mesh"
			}],

			"space":{
				"discr_order": 2,
				"advanced": {
					"isoparametric": false,
					"bc_method": "sample"
				}
			},

			"boundary_conditions": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": "x^2+y^2+z^2"
				}],
				"rhs": 6
			},

			"output": {
				"reference": {
					"solution": "x^2+y^2+z^2",
					"gradient": ["2*x","2*y","2*z"]
				},
				"paraview": {
					"high_order_mesh": false
				}
			},

			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/contact/meshes/3D/simple/bar/bar-186.msh";

	State state;
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

	// state.save_vtu("debug.vtu", 1.);

	REQUIRE(fabs(state.h1_semi_err) < 1e-7);
	REQUIRE(fabs(state.l2_err) < 1e-8);
}
