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

double five_cylinders_fluid1(double x, double y)
{
	if (x > 0.5)
		x = 1 - x;
	if (y > 0.5)
		y = 1 - y;
	if (x*x+y*y < 0.1*0.1)
		return 0;
	x -= 0.5;
	y -= 0.5;
	if (x*x+y*y < 0.1*0.1)
		return 0;
	
	return 1;
}

double five_cylinders_fluid2(double x, double y)
{
	if (x > 0.5)
		x = 1 - x;
	if (y > 0.5)
		y = 1 - y;
	if (x*x+y*y < 0.1*0.1)
		return 0.1;
	x -= 0.5;
	y -= 0.5;
	if (x*x+y*y < 0.1*0.1)
		return 0.1;
	
	return 0.9;
}

double cross_elastic1(double x, double y)
{
	x = abs(x);
	y = abs(y);
	if (x > y)
		std::swap(x, y);

	if (x < 0.1)
		return 1;
	return 1e-6;
}

double cross_elastic2(double x, double y)
{
	x = abs(x);
	y = abs(y);
	if (x > y)
		std::swap(x, y);

	if (x < 0.1)
		return 0.9;
	return 0.1;
}

TEST_CASE("elastic_homo", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "square5.msh",
                "n_refs": 0,
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "discr_order": 1
        },
        "solver": {
            "linear": {
                "solver": "Eigen::PardisoLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": true
        },
        "materials": {
            "type": "LinearElasticity",
            "E": 10,
            "nu": 0.2
        }
    })"_json;

	State state(16);
	state.init_logger("", spdlog::level::level_enum::debug, false);
	state.init(in_args);

	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

    state.assemble_rhs();

    Eigen::MatrixXd homogenized_tensor;
    Eigen::MatrixXd density_mat(state.bases.size(), 1);
    Eigen::MatrixXd barycenters;
    if (state.mesh->is_volume())
        state.mesh->cell_barycenters(barycenters);
    else
        state.mesh->face_barycenters(barycenters);
    for (int e = 0; e < state.bases.size(); e++)
    {
        density_mat(e) = cross_elastic1(barycenters(e, 0), barycenters(e, 1));
    }
    // state.density.init_multimaterial(density_mat);
    state.assembler.update_lame_params_density(density_mat);
    state.homogenize_weighted_linear_elasticity(homogenized_tensor);

    std::cout << homogenized_tensor << std::endl;

    Eigen::Matrix3d reference_tensor;
    reference_tensor <<
        1.95453,    0.102267, -0.00212518,
    0.102267,     1.95453, -0.00212518,
    -0.00212518, -0.00212518,   0.0454168;

    REQUIRE((homogenized_tensor - reference_tensor).norm() / reference_tensor.norm() < 1e-6);
}

TEST_CASE("new_version_elastic", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "cross2D.msh",
                "n_refs": 0,
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "discr_order": 1
        },
        "solver": {
            "linear": {
                "solver": "Eigen::PardisoLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": true
        },
        "materials": {
            "type": "LinearElastic",
            "E": 10,
            "nu": 0.2,
            "homogenization": true
        }
    })"_json;

	State state(16);
	state.init_logger("", spdlog::level::level_enum::debug, false);
	state.init(in_args);

	state.load_mesh();

    Eigen::MatrixXd reference_tensor;
    state.homogenize_new(reference_tensor);

    Eigen::MatrixXd homogenized_tensor;
    state.homogenize_linear_elasticity(homogenized_tensor);

    std::cout << reference_tensor << "\n" << homogenized_tensor << "\n";

    REQUIRE((homogenized_tensor - reference_tensor).norm() / reference_tensor.norm() < 1e-6);
}

TEST_CASE("new_version_stokes", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "cross2D.msh",
                "n_refs": 0,
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "discr_order": 2,
            "pressure_discr_order": 1
        },
        "solver": {
            "linear": {
                "solver": "Eigen::PardisoLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": true,
            "dirichlet_boundary": [
                {
                    "id": 7,
                    "value": [0,0]
                }
            ]
        },
        "materials": {
            "type": "Stokes",
            "E": 10,
            "nu": 0.2,
            "homogenization": true
        }
    })"_json;

	State state(16);
	state.init_logger("", spdlog::level::level_enum::debug, false);
	state.init(in_args);

	state.load_mesh();

    Eigen::MatrixXd reference_tensor;
    state.homogenize_new(reference_tensor);

    Eigen::MatrixXd homogenized_tensor;
    state.homogenize_stokes(homogenized_tensor);

    std::cout << reference_tensor << "\n" << homogenized_tensor << "\n";

    REQUIRE((homogenized_tensor - reference_tensor).norm() / reference_tensor.norm() < 1e-6);
}

TEST_CASE("elastic_homo_grad", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "square5.msh",
                "n_refs": 0,
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "discr_order": 1
        },
        "solver": {
            "linear": {
                "solver": "Eigen::PardisoLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": true
        },
        "materials": {
            "type": "LinearElasticity",
            "E": 10,
            "nu": 0.2
        }
    })"_json;

	State state(16);
	state.init_logger("", spdlog::level::level_enum::debug, false);
	state.init(in_args);

	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

    state.assemble_rhs();

    Eigen::MatrixXd homogenized_tensor;
    Eigen::MatrixXd density_mat(state.bases.size(), 1);
    Eigen::MatrixXd barycenters;
    if (state.mesh->is_volume())
        state.mesh->cell_barycenters(barycenters);
    else
        state.mesh->face_barycenters(barycenters);
    for (int e = 0; e < state.bases.size(); e++)
    {
        density_mat(e) = cross_elastic1(barycenters(e, 0), barycenters(e, 1));
    }
    // state.density.init_multimaterial(density_mat);
    state.assembler.update_lame_params_density(density_mat);

    Eigen::MatrixXd grad;
    state.homogenize_weighted_linear_elasticity_grad(homogenized_tensor, grad);
    Eigen::VectorXd trace_grad = grad.rowwise().sum();

    // finite difference
    Eigen::MatrixXd homogenized_tensor1, homogenized_tensor2;
    Eigen::MatrixXd theta(state.bases.size(), 1);
    for (int i = 0; i < theta.size(); i++)
        theta(i) = (rand() % 1000 ) / 1000.0;
    const double dt = 1e-8;

    // state.density.init_multimaterial(density_mat + theta * dt);
    state.assembler.update_lame_params_density(density_mat + theta * dt);
    state.homogenize_weighted_linear_elasticity(homogenized_tensor1);

    // state.density.init_multimaterial(density_mat - theta * dt);
    state.assembler.update_lame_params_density(density_mat - theta * dt);
    state.homogenize_weighted_linear_elasticity(homogenized_tensor2);

    const double finite_diff = (homogenized_tensor1.trace() - homogenized_tensor2.trace()) / dt / 2;
    const double analytic = (trace_grad.array() * theta.array()).sum();

    std::cout << "Finite Diff: " << finite_diff << ", analytic: " << analytic << "\n";
    REQUIRE(fabs((analytic - finite_diff) / std::max(finite_diff, analytic)) < 1e-3);
}

TEST_CASE("stokes_homo", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "2D.mesh",
                "n_refs": 0,
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "discr_order": 1,
            "pressure_discr_order": 1,
            "advanced": {
                "quadrature_order": 2
            }
        },
        "solver": {
            "linear": {
                "solver": "Eigen::PardisoLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": true
        },
        "materials": {
            "type": "Stokes",
            "viscosity": 1,
            "delta2": 0.00005,
            "use_avg_pressure": false,
            "solid_permeability": 1e-8
        }
    })"_json;

	State state(16);
	state.init_logger("", spdlog::level::level_enum::debug, false);
	state.init(in_args);

	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

    state.assemble_rhs();

    Eigen::MatrixXd homogenized_tensor;
    Eigen::MatrixXd density_mat(state.bases.size(), 1);
    Eigen::MatrixXd barycenters;
    if (state.mesh->is_volume())
        state.mesh->cell_barycenters(barycenters);
    else
        state.mesh->face_barycenters(barycenters);
    for (int e = 0; e < state.bases.size(); e++)
    {
        density_mat(e) = 1 - five_cylinders_fluid1(barycenters(e, 0), barycenters(e, 1));
    }
    // state.density.init_multimaterial(density_mat);
    state.assembler.update_lame_params_density(density_mat);
    state.homogenize_weighted_stokes(homogenized_tensor);

    std::cout << homogenized_tensor << std::endl;

    Eigen::Matrix3d reference_tensor;
    reference_tensor <<
    0.0271293, 1.40512e-12, 9.04009e-15,
    1.40508e-12,    0.027081, 1.56278e-13,
    9.04009e-15, 1.56278e-13,   0.0535809;

    REQUIRE((homogenized_tensor - reference_tensor).norm() / reference_tensor.norm() < 1e-6);
}

TEST_CASE("stokes_homo_grad", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "square6.msh",
                "n_refs": 0,
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "discr_order": 2,
            "pressure_discr_order": 1,
            "advanced": {
                "quadrature_order": 5
            }
        },
        "solver": {
            "linear": {
                "solver": "Eigen::PardisoLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": true
        },
        "materials": {
            "type": "Stokes",
            "viscosity": 1,
            "delta2": 0,
            "use_avg_pressure": true,
            "solid_permeability": 1e-8
        }
    })"_json;

	State state(16);
	state.init_logger("", spdlog::level::level_enum::debug, false);
	state.init(in_args);

	state.load_mesh();

	state.compute_mesh_stats();
	state.build_basis();

    state.assemble_rhs();

    Eigen::MatrixXd homogenized_tensor;
    Eigen::MatrixXd density_mat(state.bases.size(), 1);
    Eigen::MatrixXd barycenters;
    if (state.mesh->is_volume())
        state.mesh->cell_barycenters(barycenters);
    else
        state.mesh->face_barycenters(barycenters);
    for (int e = 0; e < state.bases.size(); e++)
    {
        density_mat(e) = 1 - five_cylinders_fluid2(barycenters(e, 0) + 0.5, barycenters(e, 1) + 0.5);
    }
    // state.density.init_multimaterial(density_mat);
    state.assembler.update_lame_params_density(density_mat);

    Eigen::MatrixXd grad;
    state.homogenize_weighted_stokes_grad(homogenized_tensor, grad);

    // finite difference
    Eigen::MatrixXd homogenized_tensor1, homogenized_tensor2;
    Eigen::MatrixXd theta(state.bases.size(), 1);
    for (int i = 0; i < theta.size(); i++)
        theta(i) = (rand() % 1000 ) / 1000.0;
    const double dt = 1e-6;

    // state.density.init_multimaterial(density_mat + theta * dt);
    state.assembler.update_lame_params_density(density_mat + theta * dt);
    state.homogenize_weighted_stokes(homogenized_tensor1);

    // state.density.init_multimaterial(density_mat - theta * dt);
    state.assembler.update_lame_params_density(density_mat - theta * dt);
    state.homogenize_weighted_stokes(homogenized_tensor2);

    const auto finite_diff = (homogenized_tensor1 - homogenized_tensor2) / dt / 2;
    Eigen::MatrixXd analytic(2, 2);
    for (int d1 = 0; d1 < state.mesh->dimension(); d1++)
        for (int d2 = 0; d2 < state.mesh->dimension(); d2++)
            analytic(d1, d2) = grad.col(d1 * state.mesh->dimension() + d2).dot(Eigen::VectorXd(theta));

    std::cout << "Finite Diff: " << finite_diff << ", analytic: " << analytic << "\n";
    REQUIRE(fabs((analytic - finite_diff).norm() / std::max(finite_diff.norm(), analytic.norm())) < 1e-3);
}
