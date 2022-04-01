// ////////////////////////////////////////////////////////////////////////////////
// #include <polyfem/State.hpp>
// #include <polyfem/auto_p_bases.hpp>
// #include <polyfem/auto_q_bases.hpp>

// #include <iostream>
// #include <fstream>
// #include <cmath>

// #include <Eigen/Dense>
// #include <unsupported/Eigen/SparseExtra>

// #include <polyfem/MaybeParallelFor.hpp>

// #include <catch2/catch.hpp>

// #include <math.h>
// ////////////////////////////////////////////////////////////////////////////////

// using namespace polyfem;

// TEST_CASE("ncmesh2d", "[ncmesh]")
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
// 					"value": "x^2+y^2"
// 				}],
//                 "exact": "x^2+y^2",
//                 "exact_grad": ["2*x","2*y"],
// 				"rhs": 4
// 			},
// 			"vismesh_rel_area": 1.0
// 		}
// 	)"_json;
//     std::string mesh_path = path + "/contact/meshes/2D/simple/circle/circle36.obj";

// 	State state(8);
// 	state.init_logger("", 6, false);
// 	state.init(in_args);
	
//     std::shared_ptr<ncMesh> ncmesh = std::make_shared<ncMesh2D>(mesh_path);

//     int id = 0;
//     int n = 0;
//     do {
//         id = rand() % ncmesh->elements.size();
//         if (ncmesh->elements[id].is_valid())
//         {
//             ncmesh->refineElement(id);
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

// 	REQUIRE(fabs(state.h1_semi_err) < 1e-9);
//     REQUIRE(fabs(state.l2_err) < 1e-10);
// }


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
//             ncmesh->refineElement(id);
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
