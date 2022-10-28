////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>

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

TEST_CASE("shape-trajectory-surface-opt", "[optimization]")
{

	const std::string path = POLYFEM_DATA_DIR;
	json target_args = R"(
        {
            "geometry": [
                {
                    "mesh": "../spline_3.obj",
                    "transformation": {
                        "translation": [
                            0,
                            0
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
                            "position": -0.5,
                            "relative": false
                        },
                        {
                            "id": 4,
                            "relative": false,
                            "box": [
                                [
                                    -0.01,
                                    -1.01
                                ],
                                [
                                    0.51,
                                    1.01
                                ]
                            ]
                        }
                    ]
                },
                {
                    "mesh": "../rectangle.obj",
                    "volume_selection": 2,
                    "n_refs": 0,
                    "advanced": {
                        "normalize_mesh": false
                    },
                    "surface_selection": [
                        {
                            "id": 2,
                            "axis": "y",
                            "position": 1.7,
                            "relative": false
                        },
                        {
                            "id": 3,
                            "axis": "-y",
                            "position": -1.7,
                            "relative": false
                        },
                        {
                            "id": 5,
                            "relative": false,
                            "box": [
                                [
                                    1.1,
                                    -1.7
                                ],
                                [
                                    1.3,
                                    1.7
                                ]
                            ]
                        }
                    ]
                }
            ],
            "space": {
                "discr_order": 1,
                "advanced": {
                    "n_boundary_samples": 5,
                    "quadrature_order": 5
                }
            },
            "time": {
                "t0": 0,
                "tend": 1.0,
                "dt": 0.1
            },
            "contact": {
                "enabled": true,
                "friction_coefficient": 0.0
            },
            "solver": {
                "linear": {
                    "solver": "Eigen::PardisoLU"
                },
                "nonlinear": {
                    "f_delta": 1e-12,
                    "grad_norm": 1e-8
                },
                "contact": {
                    "barrier_stiffness": 100000000.0
                }
            },
            "boundary_conditions": {
                "dirichlet_boundary": [
                    {
                        "id": 1,
                        "value": [
                            "1.0*t",
                            0
                        ],
                        "dimension": [
                            true,
                            true
                        ]
                    },
                    {
                        "id": 2,
                        "value": [
                            0,
                            0
                        ]
                    },
                    {
                        "id": 3,
                        "value": [
                            0,
                            0
                        ]
                    }
                ]
            },
            "materials": {
                "E": 1000000000.0,
                "nu": 0.47,
                "rho": 1000,
                "psi": 0,
                "phi": 0,
                "type": "NeoHookean"
            },
            "optimization": {
                "enabled": true
            }
        }
	)"_json;
    target_args["geometry"][0]["mesh"] = path + "/../spline_3.obj";
    target_args["geometry"][1]["mesh"] = path + "/../rectangle.obj";
	json in_args = target_args;
	in_args["geometry"][0]["mesh"] = path + "/../spline_9.obj";
	in_args["optimization"] = R"(
        {
            "enabled": true,
            "parameters": [
                {
                    "type": "shape",
                    "surface_selection": [
                        4
                    ],
                    "restriction": "cubic_hermite_spline",
                    "spline_specification": [
                        {
                            "id": 4,
                            "control_point": [
                                [
                                    0,
                                    -1
                                ],
                                [
                                    0,
                                    1
                                ]
                            ],
                            "tangent": [
                                [
                                    2,
                                    2
                                ],
                                [
                                    -2,
                                    2
                                ]
                            ],
                            "sampling": 30
                        }
                    ],
                    "smoothing_parameters": {
                        "min_iter": 2,
                        "tol": 1e-10,
                        "soft_p": 1e5,
                        "exp_factor": 5
                    }
                }
            ],
            "functionals": [
                {
                    "type": "trajectory",
                    "matching": "exact",
                    "path": "",
                    "volume_selection": [
                        2
                    ]
                }
            ],
            "solver": {
                "nonlinear": {
                    "solver": "lbfgs",
                    "min_step_size": 1e-8,
                    "max_step_size": 1.0,
                    "max_iterations": 2,
                    "grad_norm": 1e-4,
                    "f_delta": 0,
                    "line_search": {
                        "method": "backtracking",
                        "use_grad_norm_tol": 0
                    },
                    "relative_gradient": false,
                    "use_grad_norm": true,
                    "solver_info_log": true
                },
                "contact": {
                    "enabled": false
                }
            }
        }
	)"_json;
	in_args["optimization"]["solver"]["nonlinear"]["export_energy"] = "shape-trajectory-surface-opt";

	State target_state;
	target_state.init_logger("", spdlog::level::level_enum::err, false);
	target_state.init(target_args, false);
	target_state.load_mesh();
	target_state.solve();

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);
	state.load_mesh();
	state.stats.compute_mesh_stats(*state.mesh);
	state.build_basis();
	state.assemble_rhs();
	state.assemble_stiffness_mat();

	std::shared_ptr<CompositeFunctional> func = CompositeFunctional::create("Trajectory");
	auto &f = *dynamic_cast<TrajectoryFunctional *>(func.get());
	f.set_interested_ids({2}, {});
	f.set_reference(&target_state, state, {2});

	CHECK_THROWS_WITH(general_optimization(state, func), Catch::Matchers::Contains("Reached iteration limit"));

	std::ifstream energy_out("shape-trajectory-surface-opt");
	std::vector<double> energies;
	std::string line;
	if (energy_out.is_open())
	{
		while (getline(energy_out, line))
		{
			energies.push_back(std::stod(line.substr(0, line.find(","))));
		}
	}
	double starting_energy = energies[0];
	double optimized_energy = energies[energies.size() - 1];

	std::cout << starting_energy << std::endl;
	std::cout << optimized_energy << std::endl;

	REQUIRE(optimized_energy == Approx(0.6 * starting_energy).epsilon(0.05));
}
