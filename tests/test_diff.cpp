////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>
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

void vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat)
{
	int size = sqrt(vec.size());
	assert(size * size == vec.size());

	mat.resize(size, size);
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			mat(i, j) = vec(i * size + j);
}

TEST_CASE("laplacian-j(grad u)", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
{
    "geometry": [
        {
            "mesh": "3rdparty/data/circle2.msh"
        }
    ],
    "space": {
        "discr_order": 1
    },
    "boundary_conditions": {
        "rhs": -20,
        "dirichlet_boundary": [
            {
                "id": "all",
                "value": 0
            }
        ]
    },
	"materials": {
		"type": "Laplacian"
	}
}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../circle2.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

	state.assemble_stiffness_mat();
	state.assemble_rhs();

	state.solve_problem();

	IntegrableFunctional j;
	j.set_type(false, false, true);
	{
		auto j_func = [](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val = (grad_u.array() * grad_u.array()).rowwise().sum();
		};

		auto grad_j_func = [](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val = 2 * grad_u;
		};

		j.set_j(j_func);
		j.set_dj_dgradu(grad_j_func);
	}

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = vel(i) * cos(vel(i));
		}
		return vel;
	};
	double functional_val = state.J_static(j);

	Eigen::MatrixXd velocity_discrete;
	state.sample_field(velocity, velocity_discrete);

	Eigen::VectorXd one_form;
	state.dJ_shape(j, one_form);
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	// Check that the answer given is correct via finite difference.
	// First alter the mesh according to the velocity.
	const double t = 1e-6;
	state.perturb_mesh(velocity_discrete * t);

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	state.solve_problem();

	double new_functional_val = state.J_static(j);

	double finite_difference = (new_functional_val - functional_val) / t;

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("linear_elasticity-surface-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"problem": "GenericTensor",
			"scalar_formulation": "LinearElasticity",
			"n_refs": 0,
			"discr_order": 1,
			"quadrature_order": 5,
			"n_boundary_samples": 5,
			"iso_parametric": false,
			"problem_params": {
				"dirichlet_boundary": [{
					"id": 11,
					"value": [0, 0, 0]
				}],
				"rhs": [0, 10, 20]
			},
			"normalize_mesh": true,

			"boundary_sidesets": [{
				"id": 11,
				"axis": "-x",
				"position": 1e-3
			}]
		}
	)"_json;
	in_args["mesh"] = path + "/../../cube.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

	state.assemble_stiffness_mat();
	state.assemble_rhs();

	state.solve_problem();

	IntegrableFunctional j(true);
	{
		j.set_type(true, true, false);
		const auto formulation = state.formulation();
		auto j_func = [formulation](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);
			for (int i = 0; i < val.rows(); i++)
			{
				val(i) = pts(i, 0) + u(i, 0);
			}
		};

		auto dj_dx = [formulation](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			for (int i = 0; i < val.rows(); i++)
			{
				val(i, 0) = 1;
			}
		};

		j.set_j(j_func);
		j.set_dj_dx(dj_dx);
		j.set_dj_du(dj_dx);
	}

	auto velocity = [](const Eigen::MatrixXd &position) {
		Eigen::MatrixXd vel;
		vel.setZero(position.rows(), position.cols());
		for (int i = 0; i < vel.rows(); i++)
		{
			vel(i, 0) = position(i, 0);
			vel(i, 1) = position(i, 0) * position(i, 0);
			vel(i, 2) = position(i, 0);
		}
		return vel;
	};

	double functional_val = state.J_static(j);

	Eigen::MatrixXd velocity_discrete;
	state.sample_field(velocity, velocity_discrete);

	Eigen::VectorXd one_form;
	state.dJ_shape(j, one_form);
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	// Check that the answer given is correct via finite difference.
	// First alter the mesh according to the velocity.
	const double t = 1e-7;
	state.perturb_mesh(velocity_discrete * t);

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	state.solve_problem();

	double new_functional_val = state.J_static(j);

	double finite_difference = (new_functional_val - functional_val) / t;

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("linear_elasticity-surface", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"problem": "GenericTensor",
			"scalar_formulation": "LinearElasticity",
			"n_refs": 0,
			"discr_order": 1,
			"quadrature_order": 5,
			"n_boundary_samples": 5,
			"iso_parametric": false,
			"problem_params": {
				"dirichlet_boundary": [{
					"id": 11,
					"value": [0, 0]
				}],
				"rhs": [0, 10]
			},
			"normalize_mesh": true,

			"boundary_sidesets": [{
				"id": 11,
				"axis": "-x",
				"position": 1e-3
			}]
		}
	)"_json;
	in_args["mesh"] = path + "/../../cube_dense.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

	state.assemble_stiffness_mat();
	state.assemble_rhs();

	state.solve_problem();

	IntegrableFunctional j(true);
	{
		j.set_type(true, true, false);
		const auto formulation = state.formulation();
		auto j_func = [formulation](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);
			for (int i = 0; i < val.rows(); i++)
			{
				val(i) = pts(i, 0) + u(i, 0);
			}
		};

		auto dj_dx = [formulation](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			for (int i = 0; i < val.rows(); i++)
			{
				val(i, 0) = 1;
			}
		};

		j.set_j(j_func);
		j.set_dj_dx(dj_dx);
		j.set_dj_du(dj_dx);
	}

	auto velocity = [](const Eigen::MatrixXd &position) {
		Eigen::MatrixXd vel;
		vel.setZero(position.rows(), position.cols());
		for (int i = 0; i < vel.rows(); i++)
		{
			vel(i, 0) = position(i, 0);
			vel(i, 1) = position(i, 0) * position(i, 0);
		}
		return vel;
	};

	double functional_val = state.J_static(j);

	Eigen::MatrixXd velocity_discrete;
	state.sample_field(velocity, velocity_discrete);

	Eigen::VectorXd one_form;
	state.dJ_shape(j, one_form);
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	// Check that the answer given is correct via finite difference.
	// First alter the mesh according to the velocity.
	const double t = 1e-6;
	state.perturb_mesh(velocity_discrete * t);

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	state.solve_problem();

	double new_functional_val = state.J_static(j);

	double finite_difference = (new_functional_val - functional_val) / t;

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("topology-compliance", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"geometry": [
			{
				"mesh": ""
			}
		],
		"space": {
			"discr_order": 1
		},
		"solver": {
			"linear": {
				"solver": "Eigen::SparseLU"
			}
		},
		"differentiable": true,
		"boundary_conditions": {
			"rhs": [
				10,
				100,
				0
			],
			"dirichlet_boundary": [
				{
					"id": "all",
					"value": [
						0.0,
						0.0,
						0.0
					]
				}
			]
		},
		"materials": {
			"type": "LinearElasticity",
			"lambda": 17284.0,
			"mu": 7407.0
		}
	}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/bar/bar792.obj";

	auto func = CompositeFunctional::create("Compliance");
	// auto func = CompositeFunctional::create("Mass");
	// auto &func_mass = *dynamic_cast<MassFunctional *>(func.get());
	// func_mass.set_max_mass(100);
	// func_mass.set_min_mass(20);

	State state;
	state.init_logger("", spdlog::level::level_enum::debug, false);
	state.init(in_args);
	state.load_mesh();
	state.compute_mesh_stats();
	state.build_basis();
	Eigen::MatrixXd density_mat = state.assembler.lame_params().density_mat_;
	density_mat.setConstant(0.5);
	state.assembler.update_lame_params_density(density_mat, 5);
	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_export_to_file = false;
	state.solution_frames.clear();
	state.solve_problem();
	state.solve_export_to_file = true;
	double functional_val = func->energy(state);

	Eigen::MatrixXd theta(state.bases.size(), 1);
	for (int e = 0; e < state.bases.size(); e++)
		theta(e) = (rand() % 1000) / 1000.0;
	
	Eigen::VectorXd one_form = func->gradient(state, "topology");
	double derivative = (one_form.array() * theta.array()).sum();

	const double t = 1e-6;

	state.assembler.update_lame_params_density(density_mat + theta * t);
	state.assemble_stiffness_mat();
	state.solve_problem();
	double next_functional_val = func->energy(state);

	state.assembler.update_lame_params_density(density_mat - theta * t);
	state.assemble_stiffness_mat();
	state.solve_problem();
	double former_functional_val = func->energy(state);

	double finite_difference = (next_functional_val - former_functional_val) / t / 2;

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-3));
}

TEST_CASE("neohookean-j(grad u)-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"geometry": [
			{
				"mesh": ""
			}
		],
		"space": {
			"discr_order": 1
		},
		"solver": {
			"nonlinear": {
				"grad_norm": 1e-14,
				"use_grad_norm": true
			}
		},
		"differentiable": true,
		"boundary_conditions": {
			"rhs": [
				10,
				100,
				0
			],
			"dirichlet_boundary": [
				{
					"id": "all",
					"value": [
						0.0,
						0.0,
						0.0
					]
				}
			]
		},
		"materials": {
			"type": "NeoHookean",
			"lambda": 17284000.0,
			"mu": 7407410.0
		}
	}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/contact/meshes/3D/creatures/bunny.msh";

	StressFunctional func;

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();
	double functional_val = func.energy(state);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 1000) / 1000.0;
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	state.sample_field(velocity, velocity_discrete);
	Eigen::VectorXd one_form = func.gradient(state, "shape");
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	const double t = 1e-6;
	state.perturb_mesh(velocity_discrete * t);

	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double next_functional_val = func.energy(state);

	state.perturb_mesh(velocity_discrete * (-2*t));

	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double former_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - former_functional_val) / t / 2;

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-3));
}

TEST_CASE("shape-contact", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"geometry": [
			{
				"mesh": "",
				"transformation": {
					"translation": [
						0,
						1.5001
					],
					"scale": 1.0
				},
				"volume_selection": 1,
				"surface_selection": [
					{
						"id": 1,
						"axis": "y",
						"position": 1.99
					},
					{
						"id": 2,
						"axis": "-y",
						"position": 0.01
					}
				],
				"advanced": {
					"normalize_mesh": false
				}
			},
			{
				"mesh": "",
				"transformation": {
					"translation": [
						0,
						0.5
					],
					"scale": 1.0
				},
				"volume_selection": 2,
				"surface_selection": [
					{
						"id": 1,
						"axis": "y",
						"position": 1.99
					},
					{
						"id": 2,
						"axis": "-y",
						"position": 0.01
					}
				],
				"advanced": {
					"normalize_mesh": false
				}
			}
		],
		"differentiable": true,
		"contact": {
			"enabled": true,
			"dhat": 0.001
		},
		"solver": {
			"contact": {
				"barrier_stiffness": 20
			}
		},
		"boundary_conditions": {
			"dirichlet_boundary": [
				{
					"id": 1,
					"value": [
						-0.1,
						0
					]
				},
				{
					"id": 2,
					"value": [
						0,
						0
					]
				}
			]
		},
		"materials": {
			"type": "NeoHookean",
			"E": 200,
			"nu": 0.3,
			"rho": 1
		}
	}
	)"_json;
	// in_args["meshes"][0]["mesh"] = "/home/arvigjoka/adjoint-polyfem/square.obj";
	// in_args["meshes"][1]["mesh"] = "/home/arvigjoka/adjoint-polyfem/square.obj";
	in_args["geometry"][0]["mesh"] = path + "/../cube_dense.msh";
	in_args["geometry"][1]["mesh"] = path + "/../cube_dense.msh";

	StressFunctional func;

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

	state.assemble_stiffness_mat();
	state.assemble_rhs();
	state.solve_problem();
	// state.pre_sol = state.sol;
	double functional_val = func.energy(state);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 10000) / 1.0e4;
		}
		return vel;
	};

	Eigen::MatrixXd velocity_discrete;
	state.sample_field(velocity, velocity_discrete);

	Eigen::VectorXd one_form = func.gradient(state, "shape");
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	// Check that the answer given is correct via finite difference.
	// First alter the mesh according to the velocity.
	const double t = 1e-6;

	state.perturb_mesh(velocity_discrete * t);
	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double next_functional_val = func.energy(state);

	state.perturb_mesh(velocity_discrete * (-2 * t));
	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double prev_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - prev_functional_val) / 2. / t;

	REQUIRE(derivative == Approx(finite_difference).epsilon(5e-4));
}

TEST_CASE("node-trajectory", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"problem": "GenericTensor",
			"tensor_formulation": "NeoHookean",
			"problem_params": {
				"dirichlet_boundary": [{
					"id": 1,
					"value": [0, 0]
				}, {
					"id": 2,
					"value": [0, 0]
				}]
			},
			"dhat": 1e-3,
			
			"barrier_stiffness": 20,
			"meshes": [{
				"mesh": "",
				"position": [0, 1.5001],
				"scale": 1.0,
				"body_id": 1,
				"boundary_id": 1
			}, {
				"mesh": "",
				"position": [0, 0.5],
				"scale": 1.0,
				"body_id": 2,
				"boundary_id": 2
			}],

			"params": {
				"E": 200,
				"nu": 0.3,
				"rho": 1
			},

			"boundary_sidesets": [{
				"id": 1,
				"axis": "y",
				"position": 1.99
			}, {
				"id": 2,
				"axis": "-y",
				"position": 0.01
			}],

			"has_collision": true,
			"normalize_mesh": false
		}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/../../cube_dense.msh";
	in_args["meshes"][1]["mesh"] = path + "/../../cube_dense.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

	NodeTrajectoryFunctional j;
	Eigen::MatrixXd targets(state.n_bases, state.mesh->dimension());
	for (int i = 0; i < targets.size(); i++)
		targets(i) = (rand() % 10) / 10.;
	j.set_target_vertex_positions(targets);

	state.assemble_stiffness_mat();
	state.assemble_rhs();
	state.solve_problem();
	double functional_val = j.energy(state);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 10000) / 1.0e4;
		}
		return vel;
	};

	Eigen::MatrixXd velocity_discrete;
	state.sample_field(velocity, velocity_discrete, 0);

	Eigen::VectorXd one_form = j.gradient(state, "material");
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	const double t = 1e-5;

	state.perturb_material(velocity_discrete * t);
	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double next_functional_val = j.energy(state);

	state.perturb_material(velocity_discrete * (-2 * t));
	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double former_functional_val = j.energy(state);

	double finite_difference = (next_functional_val - former_functional_val) / 2. / t;

	REQUIRE(derivative == Approx(finite_difference).epsilon(2e-4));
}

TEST_CASE("material-friction-damping-transient", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"problem": "GenericTensor",
		"tensor_formulation": "NeoHookean",
		"n_refs": 0,
		"discr_order": 1,
		"time_integrator": "BDF",
		"time_integrator_params": {
			"num_steps": 2
		},
		"iso_parametric": false,
		"has_collision": true,
		"differentiable": true,
		"vismesh_rel_area": 1,
		"quadrature_order": 5,
		"problem_params": {
			"dirichlet_boundary": [{
				"id": 3,
				"value": [0, 0]
			}],
			"initial_velocity": [
				{
					"id": 1,
					"value": [5, 0]
				}
			],
			"rhs": [0, 9.8],
			"is_time_dependent": true
		},
		"mu": 0.5,
		"tend": 0.4,
		"dt": 0.01,
		"barrier_stiffness": 1e5,
		"params": {
			"E": 1e6,
			"nu": 0.3,
			"rho": 1000,
			"phi": 10,
			"psi": 10
		},
		"save_time_sequence": false,
		"skip_frame": 1,
		"meshes": [{
			"mesh": "",
			"position": [0, 0],
			"scale": [3, 0.02],
			"rotation": 0,
			"body_id": 3,
			"boundary_id": 3
		}, {
			"mesh": "",
			"position": [-1.5, 0.3],
			"scale": 0.5,
			"rotation": 0,
			"body_id": 1,
			"boundary_id": 1,
			"interested": true
		}],
		"normalize_mesh": false
	}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/../../square.obj";
	in_args["meshes"][1]["mesh"] = path + "/../../circle.msh";

	// compute reference solution
	State state_reference(8);
	auto in_args_ref = in_args;
	in_args_ref["params"]["E"] = 1e5;
	in_args_ref["mu"] = 0.2;
	in_args_ref["phi"] = 1;
	in_args_ref["psi"] = 20;
	state_reference.init_logger("", spdlog::level::level_enum::info, false);
	state_reference.init(in_args_ref);
	state_reference.load_mesh();
	state_reference.solve();

	std::set<int> interested_ids;
	if (in_args.contains("meshes") && !in_args["meshes"].empty())
	{
		const auto &meshes = in_args["meshes"].get<std::vector<json>>();
		for (const auto &m : meshes)
		{
			if (m.contains("interested") && m["interested"].get<bool>())
			{
				if (!m.contains("body_id"))
				{
					logger().error("No body id in interested mesh!");
				}
				interested_ids.insert(m["body_id"].get<int>());
			}
		}
	}

	State state(8);
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();

	TrajectoryFunctional func;
	func.set_reference(&state_reference, state, {1});
	func.set_surface_integral();

	double functional_val = func.energy(state);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(state.bases.size() * 2 + 3);
	velocity_discrete.head(state.bases.size() * 2) *= 1e3;
	velocity_discrete.tail(2) *= 1e-2;

	Eigen::VectorXd one_form = func.gradient(state, "material-full");

	const double step_size = 1e-5;
	state.perturb_material(velocity_discrete * step_size);
	state.args["mu"] = state.args["mu"].get<double>() + velocity_discrete(velocity_discrete.size() - 3) * step_size;
	state.args["params"]["psi"] = state.args["params"]["psi"].get<double>() + velocity_discrete(velocity_discrete.size() - 2) * step_size;
	state.args["params"]["phi"] = state.args["params"]["phi"].get<double>() + velocity_discrete(velocity_discrete.size() - 1) * step_size;

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	state.solve_problem();
	double next_functional_val = func.energy(state);

	state.perturb_material(velocity_discrete * (-2) * step_size);
	state.args["mu"] = state.args["mu"].get<double>() - velocity_discrete(velocity_discrete.size() - 3) * step_size * 2;
	state.args["params"]["psi"] = state.args["params"]["psi"].get<double>() - velocity_discrete(velocity_discrete.size() - 2) * step_size * 2;
	state.args["params"]["phi"] = state.args["params"]["phi"].get<double>() - velocity_discrete(velocity_discrete.size() - 1) * step_size * 2;

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	state.solve_problem();
	double former_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - former_functional_val) / step_size / 2;
	double derivative = (one_form.array() * velocity_discrete.array()).sum();
	std::cout << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("shape-transient-friction", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"problem": "GenericTensor",
			"tensor_formulation": "NeoHookean",
			"n_refs": 0,
			"discr_order": 2,
			"iso_parametric": false,
			"quadrature_order": 5,
			"problem_params": {
				"dirichlet_boundary": [{
					"id": 1,
					"value": [0, 0]
				}],
				"rhs": [0, 9.810],
				"is_time_dependent": true
			},
			"dhat": 1e-3,
			

			"t0": 0,
			"tend": 0.25,
			"time_steps": 10,

			"meshes": [{
				"mesh": "",
				"position": [0, 0],
				"scale": [5, 0.02],
				"rotation": -30,
				"body_id": 1,
				"boundary_id": 1
			}, {
				"mesh": "",
				"position": [0.2600, 0.5],
				"scale": 1.0,
				"rotation": -30,
				"body_id": 2,
				"boundary_id": 2
			}],

			"params": {
				"E": 1e6,
				"nu": 0.48,
				"rho": 1000,
				"phi": 10,
				"psi": 10
			},

			"barrier_stiffness": 1e5,
			"has_collision": true,
			"differentiable": true,
			"mu": 0.2,
			"normalize_mesh": false
		}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/../../square.obj";
	in_args["meshes"][1]["mesh"] = path + "/../../circle.msh";

	StressFunctional func;

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();
	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_geom_bases * 2, 1);
	for (int i = 0; i < state.n_geom_bases; ++i)
	{
		velocity_discrete(i * 2 + 0) = rand() % 1000;
		velocity_discrete(i * 2 + 1) = rand() % 1000;
	}

	velocity_discrete.normalize();

	Eigen::VectorXd one_form = func.gradient(state, "shape");
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	// Check that the answer given is correct via finite difference.
	// First alter the mesh according to the velocity.
	const double t = 1e-6;
	state.perturb_mesh(velocity_discrete * t);

	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double next_functional_val = func.energy(state);

	state.perturb_mesh(velocity_discrete * (-2 * t));

	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double prev_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - prev_functional_val) / (2 * t);

	std::cout << "prev functional value: " << prev_functional_val << std::endl;
	std::cout << "new functional value: " << next_functional_val << std::endl;
	std::cout << "derivative: " << derivative << std::endl;
	std::cout << "finite difference: " << finite_difference << std::endl;

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("initial-contact", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"problem": "GenericTensor",
		"tensor_formulation": "NeoHookean",
		"n_refs": 0,
		"discr_order": 2,
		"time_integrator": "BDF",
		"time_integrator_params": {
			"num_steps": 2
		},
		"iso_parametric": false,
		"has_collision": true,
		"differentiable": true,
		"vismesh_rel_area": 1,
		"quadrature_order": 5,
		"problem_params": {
			"dirichlet_boundary": [{
				"id": 3,
				"value": [0, 0]
			}],
			"initial_velocity": [
				{
					"id": 1,
					"value": [5, 0]
				}
			],
			"rhs": [0, 9.8],
			"is_time_dependent": true
		},
		"mu": 0.5,
		"tend": 0.2,
		"dt": 0.01,
		"barrier_stiffness": 23216604,
		"params": {
			"E": 1e6,
			"nu": 0.3,
			"rho": 1000,
			"phi": 10,
			"psi": 10
		},
		"save_time_sequence": false,
		"skip_frame": 1,
		"meshes": [{
			"mesh": "",
			"position": [0, 0],
			"scale": [3, 0.02],
			"rotation": 0,
			"body_id": 3,
			"boundary_id": 3
		}, {
			"mesh": "",
			"position": [-1.5, 0.3],
			"scale": 0.5,
			"rotation": 0,
			"body_id": 1,
			"boundary_id": 1,
			"interested": true
		}],
		"normalize_mesh": false
	}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/../../square.obj";
	in_args["meshes"][1]["mesh"] = path + "/../../circle.msh";

	// compute reference solution
	State state_reference(8);
	auto in_args_ref = in_args;
	in_args_ref["problem_params"]["initial_velocity"][0]["value"][0] = 4;
	in_args_ref["problem_params"]["initial_velocity"][0]["value"][1] = -1;
	state_reference.init_logger("", spdlog::level::level_enum::err, false);
	state_reference.init(in_args_ref);
	state_reference.load_mesh();
	state_reference.solve();

	State state(8);
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();

	TrajectoryFunctional func;
	func.set_reference(&state_reference, state, {1});
	func.set_surface_integral();
	func.set_transient_integral_type("uniform");

	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_bases * 2, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(i * 2 + 0) = -2.;
		velocity_discrete(i * 2 + 1) = -1.;
	}

	Eigen::VectorXd one_form = func.gradient(state, "initial-velocity");

	const double step_size = 1e-5;
	state.initial_vel_update += velocity_discrete * step_size;

	state.solve_problem();
	double next_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - functional_val) / step_size;

	REQUIRE((one_form.array() * velocity_discrete.array()).sum() == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("initial-contact-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"problem": "GenericTensor",
		"tensor_formulation": "NeoHookean",
		"n_refs": 0,
		"discr_order": 1,
		"iso_parametric": false,
		"has_collision": true,
		"differentiable": true,
		"vismesh_rel_area": 1,
		"barrier_stiffness": 1e5,
		"quadrature_order": 5,
		"problem_params": {
			"dirichlet_boundary": [{
				"id": 3,
				"value": [0, 0, 0]
			}],
			"initial_velocity": [
				{
					"id": 1,
					"value": [2, -2, -2]
				}
			],
			"rhs": [0, 9.8, 0],
			"is_time_dependent": true
		},
		"mu": 0.5,
		"tend": 0.2,
		"dt": 0.02,
		"body_params": [{
			"name": "bunny",
			"id": 1,
			"E": 1e6,
			"nu": 0.3,
			"rho": 1000
		}, {
			"name": "plane",
			"id": 3,
			"E": 1e6,
			"nu": 0.3,
			"rho": 1000
		}],
		"save_time_sequence": false,
		"skip_frame": 1,
		"meshes": [{
			"mesh": "",
			"position": [0, 0, 0],
			"scale": [3, 0.02, 1],
			"body_id": 3,
			"boundary_id": 3
		}, {
			"mesh": "",
			"position": [0, 0.3, 0],
			"scale": [0.5, 0.5, 0.5],
			"body_id": 1,
			"boundary_id": 1,
			"interested": true
		}],
		"normalize_mesh": false
	}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/contact/meshes/3D/simple/cube.msh";
	in_args["meshes"][1]["mesh"] = path + "/contact/meshes/3D/simple/sphere/sphere1K.msh";

	// compute reference solution
	State state_reference(8);
	auto in_args_ref = in_args;
	in_args_ref["problem_params"]["initial_velocity"][0]["value"][0] = 0;
	in_args_ref["problem_params"]["initial_velocity"][0]["value"][2] = 0;
	state_reference.init_logger("", spdlog::level::level_enum::err, false);
	state_reference.init(in_args_ref);
	state_reference.load_mesh();
	state_reference.solve();

	State state(8);
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();

	TrajectoryFunctional func;
	func.set_transient_integral_type("final");
	func.set_reference(&state_reference, state, {1});
	func.set_volume_integral();

	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_bases * 3, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(i * 3 + 0) = -2.;
		velocity_discrete(i * 3 + 1) = -1.;
		velocity_discrete(i * 3 + 2) = 2.;
	}

	Eigen::VectorXd one_form = func.gradient(state, "initial-velocity");

	const double step_size = 1e-7;
	state.initial_vel_update += velocity_discrete * step_size;
	state.solve_problem();
	double next_functional_val = func.energy(state);

	state.initial_vel_update -= 2 * velocity_discrete * step_size;
	state.solve_problem();
	double former_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - former_functional_val) / step_size / 2.;

	REQUIRE((one_form.array() * velocity_discrete.array()).sum() == Approx(finite_difference).epsilon(2e-4));
}

TEST_CASE("material-contact-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"problem": "GenericTensor",
		"tensor_formulation": "NeoHookean",
		"n_refs": 0,
		"discr_order": 1,
		"has_collision": true,
		"differentiable": true,
		"vismesh_rel_area": 1,
		"barrier_stiffness": 5e4,
		"problem_params": {
			"dirichlet_boundary": [{
				"id": 3,
				"value": [0, 0, 0]
			},
			{
				"id": 2,
				"value": [0, "-0.5*t", 0]
			}],
			"rhs": [0, 9.8, 0],
			"is_time_dependent": true
		},
		"mu": 0.3,
		"tend": 0.08,
		"dt": 0.01,
		"params": {
			"E": 1e6,
			"nu": 0.3,
			"rho": 100
		},
		"skip_frame": 1,
		"meshes": [{
			"mesh": "",
			"position": [0, 0, 0],
			"scale": [1, 0.02, 1],
			"rotation": 0,
			"body_id": 3,
			"boundary_id": 3
		}, {
			"mesh": "",
			"position": [0, 0.56, 0],
			"scale": [1, 0.02, 1],
			"rotation": 0,
			"body_id": 3,
			"boundary_id": 2
		}, {
			"mesh": "",
			"position": [0, 0.27, 0],
			"scale": [0.5, 0.5, 0.5],
			"rotation": 0,
			"body_id": 1,
			"boundary_id": 1,
			"interested": true
		}],
		"normalize_mesh": false
	}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/contact/meshes/3D/simple/cube.msh";
	in_args["meshes"][1]["mesh"] = path + "/contact/meshes/3D/simple/cube.msh";
	in_args["meshes"][2]["mesh"] = path + "/contact/meshes/3D/simple/sphere/sphere1K.msh";

	// compute reference solution
	State state_reference(8);
	state_reference.init_logger("", spdlog::level::level_enum::err, false);
	state_reference.init(in_args);
	state_reference.args["params"]["E"] = 1e4;
	state_reference.load_mesh();
	state_reference.solve();

	State state(8);
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();

	TrajectoryFunctional func;
	func.set_reference(&state_reference, state, {1});
	func.set_volume_integral();

	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	auto velocity = [](const Eigen::MatrixXd &position) {
		Eigen::MatrixXd vel(position.rows(), 2);
		for (int i = 0; i < vel.rows(); i++)
		{
			vel(i, 0) = 1;
			vel(i, 1) = 1;
		}
		return vel * 1e3;
	};
	state.sample_field(velocity, velocity_discrete, 0);

	Eigen::VectorXd one_form = func.gradient(state, "material");
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	const double step_size = 1e-7;
	state.perturb_material(velocity_discrete * step_size);

	state.assemble_stiffness_mat();
	state.assemble_rhs();
	state.solve_problem();
	double next_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - functional_val) / step_size;

	REQUIRE(derivative == Approx(finite_difference).epsilon(2e-4));
}

TEST_CASE("shape-contact-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"problem": "GenericTensor",
		"tensor_formulation": "NeoHookean",
		"n_refs": 0,
		"discr_order": 1,
		"iso_parametric": false,
		"has_collision": true,
		"differentiable": true,
		"vismesh_rel_area": 1,
		"quadrature_order": 5,
		"barrier_stiffness": 1822920,
		"problem_params": {
			"dirichlet_boundary": [{
				"id": 3,
				"value": [0, 0, 0]
			}],
			"initial_velocity": [
				{
					"id": 1,
					"value": [2, -2, -2]
				}
			],
			"rhs": [0, 9.8, 0],
			"is_time_dependent": true
		},
		"mu": 0.3,
		"tend": 0.2,
		"dt": 0.02,
		"body_params": [{
			"name": "bunny",
			"id": 1,
			"E": 1e6,
			"nu": 0.3,
			"rho": 1000
		}, {
			"name": "plane",
			"id": 3,
			"E": 1e6,
			"nu": 0.3,
			"rho": 1000
		}],
		"save_time_sequence": false,
		"skip_frame": 1,
		"meshes": [{
			"mesh": "",
			"position": [0, 0, 0],
			"scale": [3, 0.02, 1],
			"rotation": 0,
			"body_id": 3,
			"boundary_id": 3
		}, {
			"mesh": "",
			"position": [0, 0.3, 0],
			"scale": [0.5, 0.5, 0.5],
			"rotation": 0,
			"body_id": 1,
			"boundary_id": 1,
			"interested": true
		}],
		"normalize_mesh": false
	}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/contact/meshes/3D/simple/cube.msh";
	in_args["meshes"][1]["mesh"] = path + "/contact/meshes/3D/simple/sphere/sphere1K.msh";

	StressFunctional func;

	State state(8);
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();
	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_bases * 3, 1);
	for (int i = 0; i < velocity_discrete.size(); i++)
	{
		velocity_discrete(i) = (rand() % 1000) / 1000.0;
	}

	Eigen::VectorXd one_form = func.gradient(state, "shape");
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	const double t = 1e-7;
	state.perturb_mesh(velocity_discrete * t);

	state.assemble_rhs();
	state.assemble_stiffness_mat();
	state.solve_problem();
	double next_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - functional_val) / t;

	REQUIRE((one_form.array() * velocity_discrete.array()).sum() == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("barycenter", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"problem": "GenericTensor",
		"tensor_formulation": "NeoHookean",
		"n_refs": 0,
		"discr_order": 1,
		"time_integrator": "BDF",
		"time_integrator_params": {
			"num_steps": 2
		},
		"iso_parametric": false,
		"has_collision": true,
		"differentiable": true,
		"vismesh_rel_area": 1,
		"quadrature_order": 5,
		"problem_params": {
			"dirichlet_boundary": [{
				"id": 3,
				"value": [0, 0]
			}],
			"initial_velocity": [
				{
					"id": 1,
					"value": [5, 0]
				}
			],
			"rhs": [0, 9.8],
			"is_time_dependent": true
		},
		"mu": 0.5,
		"tend": 0.2,
		"dt": 0.01,
		"barrier_stiffness": 23216604,
		"params": {
			"E": 1e6,
			"nu": 0.3,
			"rho": 1000,
			"phi": 10,
			"psi": 10
		},
		"save_time_sequence": false,
		"skip_frame": 1,
		"meshes": [{
			"mesh": "",
			"position": [0, 0],
			"scale": [3, 0.02],
			"rotation": 0,
			"body_id": 3,
			"boundary_id": 3
		}, {
			"mesh": "",
			"position": [-1.5, 0.3],
			"scale": 0.5,
			"rotation": 0,
			"body_id": 1,
			"boundary_id": 1,
			"interested": true
		}],
		"normalize_mesh": false
	}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/../../square.obj";
	in_args["meshes"][1]["mesh"] = path + "/../../circle.msh";

	// compute reference solution
	State state_reference(8);
	auto in_args_ref = in_args;
	in_args_ref["problem_params"]["initial_velocity"][0]["value"][0] = 4;
	in_args_ref["problem_params"]["initial_velocity"][0]["value"][1] = -1;
	state_reference.init_logger("", spdlog::level::level_enum::info, false);
	state_reference.init(in_args_ref);
	state_reference.load_mesh();
	state_reference.solve();

	std::set<int> interested_ids;
	if (in_args.contains("meshes") && !in_args["meshes"].empty())
	{
		const auto &meshes = in_args["meshes"].get<std::vector<json>>();
		for (const auto &m : meshes)
		{
			if (m.contains("interested") && m["interested"].get<bool>())
			{
				if (!m.contains("body_id"))
				{
					logger().error("No body id in interested mesh!");
				}
				interested_ids.insert(m["body_id"].get<int>());
			}
		}
	}

	CenterTrajectoryFunctional func;
	func.set_interested_ids(interested_ids, {});
	std::vector<Eigen::VectorXd> barycenters;
	func.get_barycenter_series(state_reference, barycenters);
	func.set_center_series(barycenters);

	State state(8);
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();
	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_bases * 2, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(i * 2 + 0) = -2.;
		velocity_discrete(i * 2 + 1) = -1.;
	}

	Eigen::VectorXd one_form = func.gradient(state, "initial-velocity");

	const double step_size = 1e-5;
	state.initial_vel_update += velocity_discrete * step_size;

	state.solve_problem();
	double next_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - functional_val) / step_size;
	double derivative = (one_form.array() * velocity_discrete.array()).sum();
	std::cout << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("barycenter-height", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"problem": "GenericTensor",
		"tensor_formulation": "NeoHookean",
		"n_refs": 0,
		"discr_order": 1,
		"time_integrator": "BDF",
		"time_integrator_params": {
			"num_steps": 2
		},
		"iso_parametric": false,
		"has_collision": true,
		"differentiable": true,
		"vismesh_rel_area": 1,
		"quadrature_order": 5,
		"problem_params": {
			"dirichlet_boundary": [{
				"id": 3,
				"value": [0, 0]
			}],
			"initial_velocity": [
				{
					"id": 1,
					"value": [5, 0]
				}
			],
			"rhs": [0, 9.8],
			"is_time_dependent": true
		},
		"mu": 0.5,
		"tend": 0.2,
		"dt": 0.01,
		"barrier_stiffness": 23216604,
		"params": {
			"E": 1e6,
			"nu": 0.3,
			"rho": 1000,
			"phi": 10,
			"psi": 10
		},
		"save_time_sequence": false,
		"skip_frame": 1,
		"meshes": [{
			"mesh": "",
			"position": [0, 0],
			"scale": [3, 0.02],
			"rotation": 0,
			"body_id": 3,
			"boundary_id": 3
		}, {
			"mesh": "",
			"position": [-1.5, 0.3],
			"scale": 0.5,
			"rotation": 0,
			"body_id": 1,
			"boundary_id": 1,
			"interested": true
		}],
		"normalize_mesh": false
	}
	)"_json;
	in_args["meshes"][0]["mesh"] = path + "/../../square.obj";
	in_args["meshes"][1]["mesh"] = path + "/../../circle.msh";

	// compute reference solution
	State state_reference(8);
	auto in_args_ref = in_args;
	in_args_ref["problem_params"]["initial_velocity"][0]["value"][0] = 4;
	in_args_ref["problem_params"]["initial_velocity"][0]["value"][1] = -1;
	state_reference.init_logger("", spdlog::level::level_enum::info, false);
	state_reference.init(in_args_ref);
	state_reference.load_mesh();
	state_reference.solve();

	std::set<int> interested_ids;
	if (in_args.contains("meshes") && !in_args["meshes"].empty())
	{
		const auto &meshes = in_args["meshes"].get<std::vector<json>>();
		for (const auto &m : meshes)
		{
			if (m.contains("interested") && m["interested"].get<bool>())
			{
				if (!m.contains("body_id"))
				{
					logger().error("No body id in interested mesh!");
				}
				interested_ids.insert(m["body_id"].get<int>());
			}
		}
	}

	CenterTrajectoryFunctional func_aux;
	func_aux.set_interested_ids(interested_ids, {});
	std::vector<Eigen::VectorXd> barycenters;
	func_aux.get_barycenter_series(state_reference, barycenters);

	CenterXYTrajectoryFunctional func;
	func.set_center_series(barycenters);

	State state(8);
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args);
	state.load_mesh();
	state.solve();
	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_bases * 2, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(i * 2 + 0) = -200.;
		velocity_discrete(i * 2 + 1) = -100.;
	}

	Eigen::VectorXd one_form = func.gradient(state, "initial-velocity");

	const double step_size = 1e-10;
	state.initial_vel_update += velocity_discrete * step_size;

	state.solve_problem();
	double next_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - functional_val) / step_size;
	double derivative = (one_form.array() * velocity_discrete.array()).sum();
	std::cout << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}