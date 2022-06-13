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
using namespace polyfem::problem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;

TEST_CASE("ncmesh2d", "[ncmesh]")
{
	//TODO UNCOMMENT ME!
	// const std::string path = POLYFEM_DATA_DIR;
	// json in_args = R"(
	// 	{
	// 		"problem": "GenericScalar",
	// 		"materials": {"type": "Laplacian"},

	// 		"space":{
	// 			"discr_order": 2,
	// 			"advanced": {
	// 				"isoparametric": false
	// 			}
	// 		},

	// 		"boundary_conditions": {
	// 			"dirichlet_boundary": [{
	// 				"id": "all",
	// 				"value": "x^2+y^2"
	// 			}],
	// 			"rhs": 4
	// 		},

	// 		"output": {
	// 			"reference": {
	//             	"solution": "x^2+y^2",
	//             	"gradient": ["2*x","2*y"]
	// 			}
	// 		}
	// 	}
	// )"_json;
	// in_args["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";

	// State state(8);
	// state.init_logger("", 6, false);
	// state.init(in_args);

	// state.load_mesh(true);
	// NCMesh2D &ncmesh = *dynamic_cast<NCMesh2D *>(state.mesh.get());

	// for (int n = 0; n < 1; n++)
	// {
	// 	ncmesh.prepare_mesh();
	// 	std::vector<int> ref_ids(ncmesh.n_faces() / 2);
	// 	for (int i = 0; i < ref_ids.size(); i++)
	// 		ref_ids[i] = 2 * i;

	// 	ncmesh.refine_elements(ref_ids);
	// }

	// state.compute_mesh_stats();
	// state.build_basis();

	// state.assemble_stiffness_mat();
	// state.assemble_rhs();

	// state.solve_problem();
	// state.compute_errors();

	// REQUIRE(fabs(state.h1_semi_err) < 1e-9);
	// REQUIRE(fabs(state.l2_err) < 1e-10);
}

// TEST_CASE("ncmesh3d", "[ncmesh]")
// {
// 	const std::string path = POLYFEM_DATA_DIR;
// 	json in_args = R"(
// 		{
// 			"problem": "GenericScalar",
// 			"scalar_formulation": "Laplacian",
//             "export":{
//                 "high_order_mesh": false
//             },
// 			"n_refs": 0,
// 			"discr_order": 2,
// 			"iso_parametric": false,
//             "bc_method": "sample",
// 			"problem_params": {
// 				"dirichlet_boundary": [{
// 					"id": "all",
// 					"value": "x^2+y^2+z^2"
// 				}],
//                 "exact": "x^2+y^2+z^2",
//                 "exact_grad": ["2*x","2*y","2*z"],
// 				"rhs": 6
// 			},
// 			"vismesh_rel_area": 1.0
// 		}
// 	)"_json;
//     std::string mesh_path = path + "/../../torus.msh";

// 	State state(8);
// 	state.init_logger("", 6, false);
// 	state.init(in_args);

//     std::shared_ptr<ncMesh> ncmesh = std::make_shared<ncMesh3D>(mesh_path);

//     int id = 0;
//     int n = 0;
//     do {
//         id = rand() % ncmesh->elements.size();
//         if (ncmesh->elements[id].is_valid())
//         {
//             ncmesh->refine_element(id);
//             n++;
//         }
//     } while (n < 20);

//     state.load_ncMesh(ncmesh);
// 	state.compute_mesh_stats();
// 	state.build_basis();

// 	state.assemble_stiffness_mat();
// 	state.assemble_rhs();

// 	state.solve_problem();
//     state.compute_errors();

// 	REQUIRE(fabs(state.h1_semi_err) < 1e-8);
//     REQUIRE(fabs(state.l2_err) < 1e-9);
// }
