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

namespace {

void vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat)
{
	int size = sqrt(vec.size());
	assert(size * size == vec.size());

	mat.resize(size, size);
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			mat(i, j) = vec(i * size + j);
}

std::shared_ptr<State> create_state_and_solve(const json &args)
{
	std::shared_ptr<State> state = std::make_shared<State>();
	state->init_logger("", spdlog::level::level_enum::err, false);
	state->init(args, false);
	state->load_mesh();
	Eigen::MatrixXd sol, pressure;
	state->build_basis();
	state->assemble_rhs();
	state->assemble_stiffness_mat();
	state->solve_problem(sol, pressure);

	return state;
}

void solve_pde(std::shared_ptr<State> &state)
{
	state->assemble_rhs();
	state->assemble_stiffness_mat();
	Eigen::MatrixXd sol, pressure;
	state->solve_problem(sol, pressure);
}

void solve_pde(State &state)
{
	state.assemble_rhs();
	state.assemble_stiffness_mat();
	Eigen::MatrixXd sol, pressure;
	state.solve_problem(sol, pressure);
}

} // namespace

TEST_CASE("laplacian-j(grad u)", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"geometry": [
				{
					"mesh": "3rdparty/data/circle2.msh",
					"surface_selection": 1
				}
			],
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				}
			},
			"space": {
				"discr_order": 1
			},
			"boundary_conditions": {
				"rhs": [-20],
				"dirichlet_boundary": [
					{
						"id": 1,
						"value": 0
					}
				]
			},
			"materials": {
				"type": "Laplacian"
			},
			"optimization": { "enabled": true }
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../circle2.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);
	state.load_mesh();

	state.build_basis();

	solve_pde(state);

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

	solve_pde(state);

	double new_functional_val = state.J_static(j);

	double finite_difference = (new_functional_val - functional_val) / t;

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("linear_elasticity-surface-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"geometry": [
				{
					"mesh": "",
					"volume_selection": 1,
					"surface_selection": [
						{
							"id": 11,
							"axis": "-x",
							"position": 0.001
						}
					],
					"transformation": {
						"translation": [0.5, 0.5, 0.5]
					},
					"n_refs": 0
				}
			],
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				}
			},
			"space": {
				"discr_order": 1,
				"advanced": {
					"n_boundary_samples": 5,
					"quadrature_order": 5
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					10,
					20
				],
				"dirichlet_boundary": [
					{
						"id": 11,
						"value": [
							0,
							0,
							0
						]
					}
				]
			},
			"materials": {
				"type": "LinearElasticity",
				"lambda": 17284000.0,
				"mu": 7407410.0
			},
			"optimization": { "enabled": true }
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../cube.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);
	state.load_mesh();

	state.build_basis();

	solve_pde(state);

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

	solve_pde(state);

	double new_functional_val = state.J_static(j);

	double finite_difference = (new_functional_val - functional_val) / t;

	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x+dt) " << new_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("linear_elasticity-surface", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"geometry": [
				{
					"mesh": "",
					"transformation": {
						"translation": [0.5, 0.5]
					},
					"surface_selection": {
						"threshold": 1e-4
					}
				}
			],
			"space": {
				"discr_order": 1,
				"advanced": {
					"n_boundary_samples": 5,
					"quadrature_order": 5
				}
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					10
				],
				"dirichlet_boundary": [
					{
						"id": 1,
						"value": [
							0,
							0
						]
					}
				]
			},
			"materials": {
				"type": "LinearElasticity",
				"lambda": 17284000.0,
				"mu": 7407410.0
			},
			"optimization": { "enabled": true }
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../cube_dense.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);
	state.load_mesh();

	state.build_basis();

	solve_pde(state);

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

	solve_pde(state);

	double new_functional_val = state.J_static(j);

	state.perturb_mesh(velocity_discrete * (-2 * t));

	solve_pde(state);

	double old_functional_val = state.J_static(j);

	double finite_difference = (new_functional_val - old_functional_val) / t / 2;
	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << old_functional_val << " f(x+dt) " << new_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("topology-compliance", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"geometry": [
			{
				"mesh": "",
				"surface_selection": 1
			}
		],
		"space": {
			"discr_order": 1
		},
		"solver": {
			"linear": {
				"solver": "Eigen::SimplicialLDLT"
			}
		},
		"optimization": { "enabled": true },
		"boundary_conditions": {
			"rhs": [
				10,
				100,
				0
			],
			"dirichlet_boundary": [
				{
					"id": 1,
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
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);
	state.load_mesh();
	state.build_basis();
	Eigen::MatrixXd density_mat;
	density_mat.setConstant(state.mesh->n_elements(), 1, 0.5);
	state.assembler.update_lame_params_density(density_mat, 5);
	solve_pde(state);
	double functional_val = func->energy(state);

	Eigen::MatrixXd theta(state.bases.size(), 1);
	for (int e = 0; e < state.bases.size(); e++)
		theta(e) = (rand() % 1000) / 1000.0;

	Eigen::VectorXd one_form = func->gradient(state, "topology");
	double derivative = (one_form.array() * theta.array()).sum();

	const double t = 1e-6;

	state.assembler.update_lame_params_density(density_mat + theta * t);
	solve_pde(state);
	double next_functional_val = func->energy(state);

	state.assembler.update_lame_params_density(density_mat - theta * t);
	solve_pde(state);
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
				"mesh": "",
				"surface_selection": 1
			}
		],
		"space": {
			"discr_order": 1
		},
		"solver": {
			"linear": {
				"solver": "Eigen::SimplicialLDLT"
			}
		},
		"optimization": { "enabled": true },
		"boundary_conditions": {
			"rhs": [
				10,
				100,
				0
			],
			"dirichlet_boundary": [
				{
					"id": 1,
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
	state.init(in_args, false);
	state.load_mesh();

	state.build_basis();

	solve_pde(state);
	// state.pre_sol = state.diff_cached[0].u;
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

	solve_pde(state);
	double next_functional_val = func.energy(state);

	state.perturb_mesh(velocity_discrete * (-2 * t));

	solve_pde(state);
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
				]
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
				]
			}
		],
		"optimization": { "enabled": true },
		"contact": {
			"enabled": true,
			"dhat": 0.001
		},
		"solver": {
			"linear": {
				"solver": "Eigen::SimplicialLDLT"
			},
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
	in_args["geometry"][0]["mesh"] = path + "/../cube_dense.msh";
	in_args["geometry"][1]["mesh"] = path + "/../cube_dense.msh";

	StressFunctional func;

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);
	state.load_mesh();

	state.build_basis();

	solve_pde(state);
	state.pre_sol = state.diff_cached[0].u;
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
	solve_pde(state);
	double next_functional_val = func.energy(state);

	state.perturb_mesh(velocity_discrete * (-2 * t));
	solve_pde(state);
	double prev_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - prev_functional_val) / 2. / t;

	REQUIRE(derivative == Approx(finite_difference).epsilon(5e-4));
}

TEST_CASE("node-trajectory", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"geometry": [
				{
					"mesh": "",
					"volume_selection": 1,
					"surface_selection": [
						{
							"id": 1,
							"axis": "y",
							"position": 0.499
						},
						{
							"id": 2,
							"axis": "-y",
							"position": -0.499
						}
					]
				}
			],
			"contact": {
				"enabled": true,
				"dhat": 0.001
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				},
				"contact": {
					"barrier_stiffness": 20
				}
			},
			"boundary_conditions": {
				"dirichlet_boundary": [
					{
						"id": 1,
						"value": [
							0,
							0
						]
					},
					{
						"id": 2,
						"value": [
							0.2,
							0
						]
					}
				]
			},
			"materials": {
				"type": "LinearElasticity",
				"E": 200,
				"nu": 0.3,
				"rho": 1
			},
			"optimization": { "enabled": true },
			"output": {
				"paraview": {
					"vismesh_rel_area": 1,
					"skip_frame": 1
				},
				"advanced": {
					"save_time_sequence": false
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../cube_dense.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);
	state.load_mesh();

	state.build_basis();

	NodeTrajectoryFunctional j;
	Eigen::MatrixXd targets(state.n_bases, state.mesh->dimension());
	for (int i = 0; i < targets.size(); i++)
		targets(i) = (rand() % 10) / 10.;
	j.set_target_vertex_positions(targets);

	solve_pde(state);
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
	solve_pde(state);
	double next_functional_val = j.energy(state);

	state.perturb_material(velocity_discrete * (-2 * t));
	solve_pde(state);
	double former_functional_val = j.energy(state);

	double finite_difference = (next_functional_val - former_functional_val) / 2. / t;

	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("damping-transient", "[adjoint_method]")
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
							0
						],
						"rotation": 0,
						"scale": [
							3,
							0.02
						]
					},
					"volume_selection": 3,
					"surface_selection": 3,
					"n_refs": 0,
					"advanced": {
						"normalize_mesh": false
					}
				},
				{
					"mesh": "",
					"transformation": {
						"translation": [
							-1.5,
							0.3
						],
						"rotation": 0,
						"scale": 0.5
					},
					"volume_selection": 1,
					"surface_selection": 1,
					"n_refs": 0,
					"advanced": {
						"normalize_mesh": false
					}
				}
			],
			"space": {
				"discr_order": 1,
				"advanced": {
					"quadrature_order": 5
				}
			},
			"time": {
				"tend": 0.4,
				"dt": 0.01,
				"integrator": {
					"type": "BDF",
					"steps": 2
				}
			},
			"contact": {
				"enabled": true,
				"friction_coefficient": 0.5
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				},
				"contact": {
					"barrier_stiffness": 100000.0
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					9.8
				],
				"dirichlet_boundary": [
					{
						"id": 3,
						"value": [
							0,
							0
						]
					}
				]
			},
			"initial_conditions": {
				"velocity": [
					{
						"id": 1,
						"value": [
							5,
							0
						]
					}
				]
			},
			"optimization": { "enabled": true },
			"materials": {
				"type": "NeoHookean",
				"E": 1000000.0,
				"nu": 0.3,
				"rho": 1000,
				"phi": 10,
				"psi": 10
			},
			"output": {
				"paraview": {
					"vismesh_rel_area": 1,
					"skip_frame": 1
				},
				"advanced": {
					"save_time_sequence": true
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../square.obj";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	// compute reference solution
	auto in_args_ref = in_args;
	in_args_ref["materials"]["phi"] = 1;
	in_args_ref["materials"]["psi"] = 20;
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::shared_ptr<State> state = create_state_and_solve(in_args);

	TrajectoryFunctional func;
	func.set_interested_ids({1}, {});
	func.set_reference(state_reference.get(), *state, {1, 3});
	func.set_surface_integral();

	double functional_val = func.energy(*state);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(2);

	Eigen::VectorXd one_form = func.gradient(*state, "damping-parameter");

	const double step_size = 1e-5;

	state->args["materials"]["psi"] = state->args["materials"]["psi"].get<double>() + velocity_discrete(0) * step_size;
	state->args["materials"]["phi"] = state->args["materials"]["phi"].get<double>() + velocity_discrete(1) * step_size;
	state->set_materials();
	solve_pde(state);
	double next_functional_val = func.energy(*state);

	state->args["materials"]["psi"] = state->args["materials"]["psi"].get<double>() - velocity_discrete(0) * step_size * 2;
	state->args["materials"]["phi"] = state->args["materials"]["phi"].get<double>() - velocity_discrete(1) * step_size * 2;
	state->set_materials();
	solve_pde(state);
	double former_functional_val = func.energy(*state);

	double finite_difference = (next_functional_val - former_functional_val) / step_size / 2;
	double derivative = (one_form.array() * velocity_discrete.array()).sum();
	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("material-transient", "[adjoint_method]")
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
							0
						],
						"rotation": 0,
						"scale": [
							3,
							0.02
						]
					},
					"volume_selection": 3,
					"surface_selection": 3
				},
				{
					"mesh": "",
					"transformation": {
						"translation": [
							-1.5,
							0.3
						],
						"rotation": 0,
						"scale": 0.5
					},
					"volume_selection": 1,
					"surface_selection": 1
				}
			],
			"space": {
				"discr_order": 1,
				"advanced": {
					"quadrature_order": 5
				}
			},
			"time": {
				"tend": 0.4,
				"dt": 0.01,
				"integrator": {
					"type": "BDF",
					"steps": 2
				}
			},
			"contact": {
				"enabled": true,
				"friction_coefficient": 0.5
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				},
				"contact": {
					"barrier_stiffness": 1e5
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					9.8
				],
				"dirichlet_boundary": [
					{
						"id": 3,
						"value": [
							0,
							0
						]
					}
				]
			},
			"initial_conditions": {
				"velocity": [
					{
						"id": 1,
						"value": [
							5,
							0
						]
					}
				]
			},
			"optimization": { "enabled": true },
			"materials": {
				"type": "NeoHookean",
				"E": 1e6,
				"nu": 0.3,
				"rho": 1000,
				"phi": 10,
				"psi": 10
			},
			"output": {
				"paraview": {
					"vismesh_rel_area": 1,
					"skip_frame": 1
				},
				"advanced": {
					"save_time_sequence": false
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../square.obj";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	// compute reference solution
	auto in_args_ref = in_args;
	in_args_ref["materials"]["E"] = 1e5;
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::shared_ptr<State> state = create_state_and_solve(in_args);

	TrajectoryFunctional func;
	func.set_interested_ids({1}, {});
	func.set_reference(state_reference.get(), *state, {1, 3});
	func.set_surface_integral();

	double functional_val = func.energy(*state);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(state->bases.size() * 2);
	velocity_discrete *= 1e3;

	Eigen::VectorXd one_form = func.gradient(*state, "material");

	const double step_size = 1e-5;
	state->perturb_material(velocity_discrete * step_size);

	solve_pde(state);
	double next_functional_val = func.energy(*state);

	state->perturb_material(velocity_discrete * (-2) * step_size);

	solve_pde(state);
	double former_functional_val = func.energy(*state);

	double finite_difference = (next_functional_val - former_functional_val) / step_size / 2;
	double derivative = (one_form.array() * velocity_discrete.array()).sum();
	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("shape-transient-friction", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"geometry": [
				{
					"mesh": "",
					"transformation": {
						"rotation": -30,
						"scale": [
							5,
							0.02
						]
					},
					"volume_selection": 1,
					"surface_selection": 1
				},
				{
					"mesh": "",
					"transformation": {
						"translation": [
							0.26,
							0.5
						],
						"rotation": -30,
						"scale": 1.0
					},
					"volume_selection": 2,
					"surface_selection": 2
				}
			],
			"space": {
				"discr_order": 2,
				"advanced": {
					"n_boundary_samples": 5,
					"quadrature_order": 5
				}
			},
			"time": {
				"t0": 0,
				"tend": 0.25,
				"time_steps": 10
			},
			"contact": {
				"enabled": true,
				"dhat": 0.001,
				"friction_coefficient": 0.2
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				},
				"contact": {
					"barrier_stiffness": 100000.0
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					9.81
				],
				"dirichlet_boundary": [
					{
						"id": 1,
						"value": [
							0,
							0
						]
					}
				]
			},
			"optimization": { "enabled": true },
			"materials": {
				"type": "NeoHookean",
				"E": 1000000.0,
				"nu": 0.48,
				"rho": 1000,
				"phi": 10,
				"psi": 10
			},
			"output": {
				"advanced": {
					"save_time_sequence": false
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../square.obj";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	StressFunctional func;

	std::shared_ptr<State> state = create_state_and_solve(in_args);
	double functional_val = func.energy(*state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state->n_geom_bases * 2, 1);
	for (int i = 0; i < state->n_geom_bases; ++i)
	{
		velocity_discrete(i * 2 + 0) = rand() % 1000;
		velocity_discrete(i * 2 + 1) = rand() % 1000;
	}

	velocity_discrete.normalize();

	Eigen::VectorXd one_form = func.gradient(*state, "shape");
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	// Check that the answer given is correct via finite difference.
	// First alter the mesh according to the velocity.
	const double t = 1e-6;
	state->perturb_mesh(velocity_discrete * t);
	solve_pde(state);
	double next_functional_val = func.energy(*state);

	state->perturb_mesh(velocity_discrete * (-2 * t));
	solve_pde(state);
	double prev_functional_val = func.energy(*state);

	double finite_difference = (next_functional_val - prev_functional_val) / (2 * t);

	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << prev_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("shape-transient-friction-sdf", "[adjoint_method]")
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
							0
						],
						"rotation": -30,
						"scale": [
							5,
							0.02
						]
					},
					"volume_selection": 1,
					"surface_selection": 1,
					"n_refs": 0,
					"advanced": {
						"normalize_mesh": false
					}
				},
				{
					"mesh": "",
					"transformation": {
						"translation": [
							0.26,
							0.5
						],
						"rotation": -30,
						"scale": 1.0
					},
					"volume_selection": 2,
					"surface_selection": [{
						"id": 2,
						"relative": false,
						"box": [
							[0.11, 0.55],
							[0.8, 1]
						]
					}],
					"n_refs": 0,
					"advanced": {
						"normalize_mesh": false
					}
				}
			],
			"space": {
				"discr_order": 2,
				"advanced": {
					"n_boundary_samples": 5,
					"quadrature_order": 5
				}
			},
			"time": {
				"t0": 0,
				"tend": 0.25,
				"time_steps": 10
			},
			"contact": {
				"enabled": true,
				"dhat": 0.001,
				"friction_coefficient": 0.2
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				},
				"contact": {
					"barrier_stiffness": 100000.0
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					9.81
				],
				"dirichlet_boundary": [
					{
						"id": 1,
						"value": [
							0,
							0
						]
					}
				]
			},
			"optimization": { "enabled": true },
			"materials": {
				"type": "NeoHookean",
				"E": 1000000.0,
				"nu": 0.48,
				"rho": 1000,
				"phi": 10,
				"psi": 10
			},
			"output": {
				"paraview": {
					"vismesh_rel_area": 1,
					"skip_frame": 1
				},
				"advanced": {
					"save_time_sequence": false
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../square.obj";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	Eigen::MatrixXd control_points, tangents, delta;
	control_points.setZero(2, 2);
	control_points << 1, 0.4,
		0.1, 1;
	tangents.setZero(2, 2);
	tangents << -1, 1,
		-1, 0;
	delta.setZero(1, 2);
	delta << 0.05, 0.05;
	SDFTrajectoryFunctional func;
	func.set_spline_target(control_points, tangents, delta);
	func.set_interested_ids({}, {2});
	func.set_surface_integral();
	func.set_transient_integral_type("final");

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State& state = *state_ptr;
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

	solve_pde(state);
	double next_functional_val = func.energy(state);

	state.perturb_mesh(velocity_discrete * (-2 * t));

	solve_pde(state);
	double prev_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - prev_functional_val) / (2 * t);

	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << prev_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("initial-contact", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"geometry": [
				{
					"mesh": "",
					"transformation": {
						"scale": [
							3,
							0.02
						]
					},
					"volume_selection": 3,
					"surface_selection": 3
				},
				{
					"mesh": "",
					"transformation": {
						"translation": [
							-1.5,
							0.3
						],
						"scale": 0.5
					},
					"volume_selection": 1,
					"surface_selection": 1
				}
			],
			"space": {
				"discr_order": 2,
				"advanced": {
					"quadrature_order": 5
				}
			},
			"time": {
				"tend": 0.2,
				"dt": 0.04,
				"integrator": {
					"type": "BDF",
					"steps": 2
				}
			},
			"contact": {
				"enabled": true,
				"friction_coefficient": 0.2
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				},
				"contact": {
					"barrier_stiffness": 1e4
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					9.8
				],
				"dirichlet_boundary": [
					{
						"id": 3,
						"value": [
							0,
							0
						]
					}
				]
			},
			"initial_conditions": {
				"velocity": [
					{
						"id": 1,
						"value": [
							5,
							0
						]
					}
				]
			},
			"optimization": { "enabled": true },
			"materials": {
				"type": "NeoHookean",
				"E": 1e5,
				"nu": 0.3,
				"rho": 1000
			},
			"output": {
				"paraview": {
					"vismesh_rel_area": 1,
					"skip_frame": 1
				},
				"advanced": {
					"save_time_sequence": false
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../square.obj";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	// compute reference solution
	auto in_args_ref = in_args;
	in_args_ref["initial_conditions"]["velocity"][0]["value"][0] = 4;
	in_args_ref["initial_conditions"]["velocity"][0]["value"][1] = -1;
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State& state = *state_ptr;

	TrajectoryFunctional func;
	func.set_reference(state_reference.get(), state, {1, 3});
	func.set_interested_ids({1}, {});
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

	solve_pde(state);
	double next_functional_val = func.energy(state);

	state.initial_vel_update -= velocity_discrete * step_size * 2;

	solve_pde(state);
	double last_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - last_functional_val) / step_size / 2;
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << last_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("barycenter", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"geometry": [
				{
					"mesh": "",
					"transformation": {
						"scale": [
							3,
							0.02
						]
					},
					"volume_selection": 3,
					"surface_selection": 3
				},
				{
					"mesh": "",
					"transformation": {
						"translation": [
							-1.5,
							0.3
						],
						"scale": 0.5
					},
					"volume_selection": 1,
					"surface_selection": 1
				}
			],
			"space": {
				"discr_order": 1,
				"advanced": {
					"quadrature_order": 5
				}
			},
			"time": {
				"tend": 0.2,
				"dt": 0.02,
				"integrator": {
					"type": "BDF",
					"steps": 2
				}
			},
			"contact": {
				"enabled": true,
				"friction_coefficient": 0.2
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				},
				"contact": {
					"barrier_stiffness": 1e5
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					9.8
				],
				"dirichlet_boundary": [
					{
						"id": 3,
						"value": [
							0,
							0
						]
					}
				]
			},
			"initial_conditions": {
				"velocity": [
					{
						"id": 1,
						"value": [
							5,
							0
						]
					}
				]
			},
			"optimization": { "enabled": true },
			"materials": {
				"type": "NeoHookean",
				"E": 1000000.0,
				"nu": 0.3,
				"rho": 1000
			},
			"output": {
				"paraview": {
					"vismesh_rel_area": 1
				},
				"advanced": {
					"save_time_sequence": false
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../square.obj";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	// compute reference solution
	auto in_args_ref = in_args;
	in_args_ref["initial_conditions"]["velocity"][0]["value"][0] = 4;
	in_args_ref["initial_conditions"]["velocity"][0]["value"][1] = -1;
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	CenterTrajectoryFunctional func;
	func.set_interested_ids({1, 3}, {});
	std::vector<Eigen::VectorXd> barycenters;
	func.get_barycenter_series(*state_reference, barycenters);
	func.set_center_series(barycenters);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State& state = *state_ptr;
	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_bases * 2, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(i * 2 + 0) = -2.;
		velocity_discrete(i * 2 + 1) = -1.;
	}

	Eigen::VectorXd one_form = func.gradient(state, "initial-velocity");

	const double step_size = 1e-6;
	state.initial_vel_update += velocity_discrete * step_size;

	solve_pde(state);
	double next_functional_val = func.energy(state);

	state.initial_vel_update -= velocity_discrete * step_size * 2;

	solve_pde(state);
	double last_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - last_functional_val) / step_size / 2;
	double derivative = (one_form.array() * velocity_discrete.array()).sum();
	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << last_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
}

TEST_CASE("barycenter-height", "[adjoint_method]")
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
							0
						],
						"rotation": 0,
						"scale": [
							3,
							0.02
						]
					},
					"volume_selection": 3,
					"surface_selection": 3,
					"n_refs": 0,
					"advanced": {
						"normalize_mesh": false
					}
				},
				{
					"mesh": "",
					"transformation": {
						"translation": [
							-1.5,
							0.3
						],
						"rotation": 0,
						"scale": 0.5
					},
					"volume_selection": 1,
					"surface_selection": 1,
					"n_refs": 0,
					"advanced": {
						"normalize_mesh": false
					}
				}
			],
			"space": {
				"discr_order": 1,
				"advanced": {
					"quadrature_order": 5
				}
			},
			"time": {
				"tend": 0.2,
				"dt": 0.01,
				"integrator": {
					"type": "BDF",
					"steps": 2
				}
			},
			"contact": {
				"enabled": true,
				"friction_coefficient": 0.5
			},
			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				},
				"contact": {
					"barrier_stiffness": 23216604
				}
			},
			"boundary_conditions": {
				"rhs": [
					0,
					9.8
				],
				"dirichlet_boundary": [
					{
						"id": 3,
						"value": [
							0,
							0
						]
					}
				]
			},
			"initial_conditions": {
				"velocity": [
					{
						"id": 1,
						"value": [
							5,
							0
						]
					}
				]
			},
			"optimization": { "enabled": true },
			"materials": {
				"type": "NeoHookean",
				"E": 1000000.0,
				"nu": 0.3,
				"rho": 1000,
				"phi": 10,
				"psi": 10
			},
			"output": {
				"paraview": {
					"vismesh_rel_area": 1
				},
				"advanced": {
					"save_time_sequence": false
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../square.obj";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	// compute reference solution
	auto in_args_ref = in_args;
	in_args_ref["initial_conditions"]["velocity"][0]["value"][0] = 4;
	in_args_ref["initial_conditions"]["velocity"][0]["value"][1] = -1;
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	CenterTrajectoryFunctional func_aux;
	func_aux.set_interested_ids({1}, {});
	std::vector<Eigen::VectorXd> barycenters;
	func_aux.get_barycenter_series(*state_reference, barycenters);

	CenterXYTrajectoryFunctional func;
	func.set_interested_ids({1}, {});
	func.set_center_series(barycenters);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State& state = *state_ptr;
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

	solve_pde(state);
	double next_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - functional_val) / step_size;
	double derivative = (one_form.array() * velocity_discrete.array()).sum();
	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("dirichlet-sdf", "[adjoint_method]")
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
						0
					],
					"rotation": 0,
					"scale": [
						2,
						1
					]
				},
				"volume_selection": 1,
				"n_refs": 0,
				"advanced": {
					"normalize_mesh": false
				},
				"surface_selection": [
					{
						"id": 1,
						"axis": "-x",
						"position": -2,
						"relative": false
					},
					{
						"id": 2,
						"axis": "x",
						"position": 2,
						"relative": false
					},
					{
						"id": 4,
						"relative": false,
						"box": [
							[
								-2,
								-1
							],
							[
								2,
								0
							]
						]
					}
				]
			},
			{
				"mesh": "",
				"transformation": {
					"translation": [
						0,
						0.5
					],
					"rotation": 0,
					"scale": 0.5
				},
				"volume_selection": 3,
				"n_refs": 0,
				"advanced": {
					"normalize_mesh": false
				},
				"surface_selection": [
					{
						"id": 3,
						"axis": "y",
						"position": 0.5,
						"relative": false
					}
				]
			}
		],
		"optimization": { "enabled": true },
		"space": {
			"discr_order": 2,
			"advanced": {
				"n_boundary_samples": 5,
				"quadrature_order": 5
			}
		},
		"time": {
			"tend": 0.2,
			"dt": 0.02,
			"integrator": {
				"type": "BDF",
				"steps": 1
			}
		},
		"contact": {
			"enabled": true,
			"friction_coefficient": 0.5
		},
		"solver": {
			"linear": {
				"solver": "Eigen::SimplicialLDLT"
			},
			"contact": {
				"barrier_stiffness": 100000
			}
		},
		"boundary_conditions": {
			"rhs": [
				0,
				9.8
			],
			"dirichlet_boundary": [
				{
					"id": 1,
					"time_reference": [
						0.02,
						0.04,
						0.06,
						0.08,
						0.1,
						0.12,
						0.14,
						0.16,
						0.18,
						0.2,
						0.22
					],
					"value": [
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						],
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						]
					]
				},
				{
					"id": 2,
					"time_reference": [
						0.02,
						0.04,
						0.06,
						0.08,
						0.1,
						0.12,
						0.14,
						0.16,
						0.18,
						0.2,
						0.22
					],
					"value": [
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						],
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						]
					]
				},
				{
					"id": 3,
					"time_reference": [
						0.02,
						0.04,
						0.06,
						0.08,
						0.1,
						0.12,
						0.14,
						0.16,
						0.18,
						0.2,
						0.22
					],
					"value": [
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						],
						[
							0.0,
							-0.08,
							-0.16,
							-0.24,
							-0.32,
							-0.4,
							-0.48,
							-0.56,
							-0.64,
							-0.72,
							-0.8
						]
					]
				}
			]
		},
		"materials": {
			"type": "NeoHookean",
			"E": 1000000.0,
			"nu": 0.3,
			"rho": 1000
		},
		"output": {
			"paraview": {
				"vismesh_rel_area": 1
			},
			"advanced": {
				"save_time_sequence": true
			}
		}
	}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../floor.msh";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State& state = *state_ptr;

	Eigen::MatrixXd control_points, tangents, delta;
	control_points.setZero(2, 2);
	control_points << -2.5, -0.1,
		2.5, -0.1;
	tangents.setZero(2, 2);
	tangents << 1.5, -2,
		1.5, 2;
	delta.setZero(1, 2);
	delta << 0.5, 0.5;
	SDFTrajectoryFunctional func;
	func.set_spline_target(control_points, tangents, delta);
	func.set_interested_ids({}, {4});
	func.set_surface_integral();
	func.set_transient_integral_type("step_10");

	const double functional_val = func.energy(state);

	int time_steps = state.args["time"]["time_steps"].get<int>();

	Eigen::VectorXd one_form = func.gradient(state, "dirichlet");
	// std::cout << "one form " << one_form << std::endl;

	// srand(time(0));

	double derivative;
	double finite_difference;

	derivative = 0;
	finite_difference = 0;

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(time_steps, state.mesh->dimension());
	for (int j = 0; j < time_steps; ++j)
	{
		for (int i = 0; i < state.mesh->dimension(); ++i)
		{
			double random_val = (rand() % 200) / 100. - 1.;
			velocity_discrete(j, i) = random_val;
		}
	}

	const double step_size = 1e-7;

	json temp_args = in_args;
	for (int i = 0; i < 2; ++i)
	{
		for (int t = 0; t < time_steps; ++t)
		{
			temp_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
			temp_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
			temp_args["boundary_conditions"]["dirichlet_boundary"][2]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][2]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
		}
	}
	std::shared_ptr<State> state_fd = create_state_and_solve(temp_args);
	double next_functional_val = func.energy(*state_fd);

	finite_difference = (next_functional_val - functional_val) / step_size;
	for (int j = 0; j < time_steps; ++j)
		for (int i = 0; i < state.boundary_nodes.size(); ++i)
			derivative += one_form(j * state.boundary_nodes.size() + i) * velocity_discrete(j, i % 2);
	std::cout << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("dirichlet-ref", "[adjoint_method]")
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
						0
					],
					"rotation": 0,
					"scale": [
						2,
						1
					]
				},
				"volume_selection": 1,
				"n_refs": 0,
				"advanced": {
					"normalize_mesh": false
				},
				"surface_selection": [
					{
						"id": 1,
						"axis": "-x",
						"position": -2,
						"relative": false
					},
					{
						"id": 2,
						"axis": "x",
						"position": 2,
						"relative": false
					},
					{
						"id": 4,
						"relative": false,
						"box": [
							[
								-2,
								-1
							],
							[
								2,
								0
							]
						]
					}
				]
			},
			{
				"mesh": "",
				"transformation": {
					"translation": [
						0,
						0.5
					],
					"rotation": 0,
					"scale": 0.5
				},
				"volume_selection": 3,
				"n_refs": 0,
				"advanced": {
					"normalize_mesh": false
				},
				"surface_selection": [
					{
						"id": 3,
						"axis": "y",
						"position": 0.5,
						"relative": false
					}
				]
			}
		],
		"optimization": { "enabled": true },
		"space": {
			"discr_order": 1,
			"advanced": {
				"n_boundary_samples": 5,
				"quadrature_order": 5
			}
		},
		"time": {
			"tend": 0.2,
			"dt": 0.02,
			"integrator": {
				"type": "BDF",
				"steps": 1
			}
		},
		"contact": {
			"enabled": true,
			"friction_coefficient": 0
		},
		"solver": {
			"linear": {
				"solver": "Eigen::SimplicialLDLT"
			},
			"contact": {
				"barrier_stiffness": 100000
			}
		},
		"boundary_conditions": {
			"rhs": [
				0,
				9.8
			],
			"dirichlet_boundary": [
				{
					"id": 1,
					"time_reference": [
						0.02,
						0.04,
						0.06,
						0.08,
						0.1,
						0.12,
						0.14,
						0.16,
						0.18,
						0.2,
						0.22
					],
					"value": [
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						],
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						]
					]
				},
				{
					"id": 2,
					"time_reference": [
						0.02,
						0.04,
						0.06,
						0.08,
						0.1,
						0.12,
						0.14,
						0.16,
						0.18,
						0.2,
						0.22
					],
					"value": [
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						],
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						]
					]
				},
				{
					"id": 3,
					"time_reference": [
						0.02,
						0.04,
						0.06,
						0.08,
						0.1,
						0.12,
						0.14,
						0.16,
						0.18,
						0.2,
						0.22
					],
					"value": [
						[
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0,
							0
						],
						[
							0.0,
							-0.08,
							-0.16,
							-0.24,
							-0.32,
							-0.4,
							-0.48,
							-0.56,
							-0.64,
							-0.72,
							-0.8
						]
					]
				}
			]
		},
		"materials": {
			"type": "NeoHookean",
			"E": 1000000.0,
			"nu": 0.3,
			"rho": 1000
		},
		"output": {
			"paraview": {
				"vismesh_rel_area": 1
			},
			"advanced": {
				"save_time_sequence": true
			}
		}
	}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/../floor.msh";
	in_args["geometry"][1]["mesh"] = path + "/../circle.msh";

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State& state = *state_ptr;

	int time_steps = state.args["time"]["time_steps"].get<int>();

	json ref_args = in_args;
	for (int t = 0; t < time_steps; ++t)
	{
		ref_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][0][t] = ref_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][0][t].get<double>() - 0.5 * t;
		ref_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][0][t] = ref_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][0][t].get<double>() + 0.5 * t;
	}
	std::shared_ptr<State> state_ref = create_state_and_solve(ref_args);

	TrajectoryFunctional func;
	func.set_reference(state_ref.get(), state, {2});
	func.set_interested_ids({}, {4});
	func.set_surface_integral();
	func.set_transient_integral_type("uniform");

	const double functional_val = func.energy(state);

	Eigen::VectorXd one_form = func.gradient(state, "dirichlet");
	// std::cout << "one form " << one_form << std::endl;

	// srand(time(0));

	double derivative;
	double finite_difference;

	derivative = 0;
	finite_difference = 0;

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(time_steps, state.mesh->dimension());
	for (int j = 0; j < time_steps; ++j)
	{
		for (int i = 0; i < state.mesh->dimension(); ++i)
		{
			double random_val = (rand() % 200) / 100. - 1.;
			velocity_discrete(j, i) = random_val;
		}
	}

	const double step_size = 1e-7;

	json temp_args = in_args;
	for (int i = 0; i < 2; ++i)
	{
		for (int t = 0; t < time_steps; ++t)
		{
			temp_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
			temp_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
			temp_args["boundary_conditions"]["dirichlet_boundary"][2]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][2]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
		}
	}
	std::shared_ptr<State> state_fd = create_state_and_solve(temp_args);
	double next_functional_val = func.energy(*state_fd);

	finite_difference = (next_functional_val - functional_val) / step_size;
	for (int j = 0; j < time_steps; ++j)
		for (int i = 0; i < state.boundary_nodes.size(); ++i)
			derivative += one_form(j * state.boundary_nodes.size() + i) * velocity_discrete(j, i % 2);
	std::cout << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}