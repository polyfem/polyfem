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
            "use_avg_pressure": false
        }
    })"_json;

	State state(16);
	state.init_logger("", 1, false);
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
        density_mat(e) = five_cylinders_fluid1(barycenters(e, 0), barycenters(e, 1));
    }
    state.density.init_multimaterial(density_mat);
    state.homogenize_weighted_stokes(homogenized_tensor);

    std::cout << homogenized_tensor << std::endl;

    Eigen::Matrix3d reference_tensor;
    reference_tensor <<
        0.0279302   , -1.00961e-16, -2.71248e-17,
        -1.02466e-16,    0.0280755,  4.53845e-16,
        -2.71248e-17,  4.53845e-16,    0.0559471;

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
	state.init_logger("", 1, false);
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
        density_mat(e) = five_cylinders_fluid2(barycenters(e, 0) + 0.5, barycenters(e, 1) + 0.5);
    }
    state.density.init_multimaterial(density_mat);

    Eigen::VectorXd grad;
    state.homogenize_weighted_stokes_grad(homogenized_tensor, grad);

    // finite difference
    Eigen::MatrixXd homogenized_tensor1, homogenized_tensor2;
    Eigen::MatrixXd theta(state.bases.size(), 1);
    for (int i = 0; i < theta.size(); i++)
        theta(i) = (rand() % 1000 ) / 1000.0;
    const double dt = 1e-6;

    state.density.init_multimaterial(density_mat + theta * dt);
    state.assemble_rhs();
    state.homogenize_weighted_stokes(homogenized_tensor1);

    state.density.init_multimaterial(density_mat - theta * dt);
    state.assemble_rhs();
    state.homogenize_weighted_stokes(homogenized_tensor2);

    const double finite_diff = (homogenized_tensor1.trace() - homogenized_tensor2.trace()) / state.mesh->dimension() / dt / 2;
    const double analytic = (grad.array() * theta.array()).sum();

    std::cout << "Finite Diff: " << finite_diff << ", analytic: " << analytic << "\n";
    REQUIRE(fabs((analytic - finite_diff) / std::max(finite_diff, analytic)) < 1e-3);
}
