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

TEST_CASE("density_elastic_homo", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "square5.msh",
                "n_refs": 0,
                "surface_selection": {
                    "threshold": 1e-6
                },
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "advanced": {
                "periodic_basis": true
            },
            "discr_order": 1
        },
        "solver": {
            "linear": {
                "solver": "Eigen::SimplicialLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": [true, true, true]
        },
        "materials": {
            "type": "LinearElasticity",
            "E": 10,
            "nu": 0.2
        }
    })"_json;
    in_args["geometry"][0]["mesh"] = path + "/../square5.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);

	state.load_mesh();

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

TEST_CASE("density_elastic_homo_grad", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "square5.msh",
                "n_refs": 0,
                "surface_selection": {
                    "threshold": 1e-6
                },
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "advanced": {
                "periodic_basis": true
            },
            "discr_order": 1
        },
        "solver": {
            "linear": {
                "solver": "Eigen::SimplicialLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": [true, true, true]
        },
        "materials": {
            "type": "LinearElasticity",
            "E": 10,
            "nu": 0.2
        }
    })"_json;
    in_args["geometry"][0]["mesh"] = path + "/../square5.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);

	state.load_mesh();

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
    // Eigen::VectorXd trace_grad = grad.col(0) + grad.col(4) + grad.col(8);

    Eigen::VectorXd random_coeff(grad.cols());
    for (int i = 0; i < random_coeff.size(); i++)
        random_coeff(i) = (rand() % 1000) / 1000.0;

    Eigen::VectorXd total_grad = grad * random_coeff;

    // finite difference
    Eigen::MatrixXd homogenized_tensor1, homogenized_tensor2;
    Eigen::MatrixXd theta(state.bases.size(), 1);
    for (int i = 0; i < theta.size(); i++)
        theta(i) = (rand() % 1000) / 1000.0;
    const double dt = 1e-8;

    // state.density.init_multimaterial(density_mat + theta * dt);
    state.assembler.update_lame_params_density(density_mat + theta * dt);
    state.homogenize_weighted_linear_elasticity(homogenized_tensor1);

    // state.density.init_multimaterial(density_mat - theta * dt);
    state.assembler.update_lame_params_density(density_mat - theta * dt);
    state.homogenize_weighted_linear_elasticity(homogenized_tensor2);
    
    Eigen::MatrixXd f_diff_mat = homogenized_tensor1 - homogenized_tensor2;
    Eigen::VectorXd f_diff(Eigen::Map<Eigen::VectorXd>(f_diff_mat.data(), f_diff_mat.cols()*f_diff_mat.rows()));

    const double finite_diff = f_diff.dot(random_coeff) / dt / 2;
    const double analytic = (total_grad.array() * theta.array()).sum();

    std::cout << "Finite Diff: " << finite_diff << ", analytic: " << analytic << "\n";
    REQUIRE(fabs((analytic - finite_diff) / std::max(finite_diff, analytic)) < 1e-3);
}

TEST_CASE("shape_elastic_homo_grad", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "",
                "n_refs": 0,
                "surface_selection": {
                    "threshold": 1e-5
                }
            }
        ],
        "space": {
            "discr_order": 1,
            "advanced": {
                "quadrature_order": 5
            }
        },
        "solver": {
            "linear": {
                "solver": "Eigen::SparseLU"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": [true, true, true]
        },
        "materials": {
            "type": "LinearElasticity",
            "E": 10,
            "nu": 0.2
        }
    })"_json;
    in_args["geometry"][0]["mesh"] = path + "/../cross2d.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);

	state.load_mesh();
	state.build_basis();

    Eigen::MatrixXd homogenized_tensor;

    Eigen::MatrixXd grad;
    state.homogenize_linear_elasticity_shape_grad(homogenized_tensor, grad);
    // Eigen::VectorXd trace_grad = grad.col(0) + grad.col(4) + grad.col(8);

    Eigen::VectorXd random_coeff(9);
    //for (int i = 0; i < random_coeff.size(); i++)
    //    random_coeff(i) = (rand() % 1000) / 1000.0;
    random_coeff << 0.383, 0.886, 0.777, 0.915, 0.793, 0.335, 0.386, 0.492, 0.649;

    Eigen::VectorXd total_grad = grad * random_coeff;

    const int dim = state.mesh->dimension();
    // finite difference
    Eigen::MatrixXd homogenized_tensor1, homogenized_tensor2;
    Eigen::MatrixXd theta(state.n_geom_bases * dim, 1);
    //for (int i = 0; i < theta.size(); i++)
    //    theta(i) = (rand() % 1000) / 1000.0;
	theta << 0.421, 0.362, 0.027, 0.69, 0.059, 0.763, 0.926, 0.54, 0.426, 0.172, 0.736, 0.211, 0.368, 0.567, 0.429, 0.782, 0.53, 0.862, 0, 0, 0.135, 0.929, 0, 0, 0.058, 0.069, 0.167, 0.393, 0, 0, 0, 0, 0.373, 0.421, 0, 0, 0.537, 0.198, 0.324, 0.315, 0, 0, 0, 0, 0.98, 0.956, 0, 0, 0.17, 0.996, 0.281, 0.305, 0, 0, 0, 0, 0.505, 0.846, 0, 0, 0.857, 0.124, 0.895, 0.582, 0, 0, 0.367, 0.434, 0.364, 0.043, 0.75, 0.087, 0.808, 0.276, 0.178, 0.788, 0.584, 0.403, 0.651, 0.754, 0.399, 0.932, 0.06, 0.676, 0.368, 0.739, 0.012, 0.226, 0.586, 0.094, 0.539, 0.795, 0.57, 0.434, 0.378, 0.467, 0.601, 0.097, 0.902, 0.317, 0.492, 0.652, 0.756, 0.301, 0.28, 0.286, 0.441, 0.865, 0.689, 0.444, 0.619, 0.44, 0.729, 0.031, 0, 0, 0.771, 0.481, 0, 0, 0, 0, 0.856, 0.497, 0, 0, 0, 0, 0.683, 0.219, 0, 0, 0, 0, 0.829, 0.503, 0, 0, 0.368, 0.708, 0.715, 0.34, 0.149, 0.796, 0.723, 0.618, 0.245, 0.846, 0.451, 0.921, 0.555, 0.379, 0.488, 0.764, 0.228, 0.841, 0.35, 0.193, 0.5, 0.034, 0.764, 0.124, 0.914, 0.987, 0.856, 0.743, 0.491, 0.227, 0.365, 0.859, 0.936, 0.432, 0.551, 0.437, 0.228, 0.275, 0.407, 0.474, 0.121, 0.858, 0.395, 0.029, 0.237, 0.235, 0.793, 0.818, 0.428, 0.143, 0.011, 0.928, 0.529, 0.776, 0.404, 0.443, 0.763, 0.613, 0.538, 0.606, 0.84, 0.904, 0.818, 0.128, 0.688, 0.369, 0.917, 0.917, 0.996, 0.324, 0.743, 0.47, 0.183, 0.49, 0.499, 0.772, 0.725, 0.644, 0.59, 0.505, 0.139, 0.954, 0.786, 0.669, 0.082, 0.542, 0.464, 0.197, 0.507, 0.355, 0.804, 0.348, 0.611, 0.622, 0.828, 0.299, 0.343, 0.746, 0.568, 0.34, 0.422, 0.311, 0.81, 0.605, 0.801, 0.661, 0.73, 0.878, 0.305, 0.32, 0.736, 0.444, 0.626, 0.522, 0.465, 0.708, 0.416, 0.282, 0.258, 0.924, 0.637, 0.062, 0.624, 0.6, 0.036, 0.452, 0.899, 0.379, 0.55, 0.468, 0.071, 0.973, 0.131, 0.881, 0.93, 0.933, 0.894, 0.66, 0.163, 0.199, 0.981, 0.899, 0.996, 0.959, 0.773, 0.813, 0.668, 0.19, 0.095, 0.926, 0.466, 0.084, 0.34, 0.09, 0.684, 0.376, 0.542, 0.936, 0.107, 0.445;

    for (int i = 0; i < state.n_geom_bases; i++)
    {
        auto node = state.geom_mesh_nodes->node_position(i);
        if (node(0) > 0.5) node(0) = 1 - node(0);
        if (node(1) > 0.5) node(1) = 1 - node(1);
        if (node.minCoeff() < 1e-3)
            theta.block(i * dim, 0, dim, 1).setZero();
    }
    const double dt = 1e-7;

    state.perturb_mesh(theta * dt);
    state.homogenization(homogenized_tensor1);

    state.perturb_mesh(theta * (-2*dt));
    state.homogenization(homogenized_tensor2);
    
    Eigen::MatrixXd f_diff_mat = homogenized_tensor1 - homogenized_tensor2;
    Eigen::VectorXd f_diff(Eigen::Map<Eigen::VectorXd>(f_diff_mat.data(), f_diff_mat.cols()*f_diff_mat.rows()));

    const double finite_diff = f_diff.dot(random_coeff) / dt / 2;
    const double analytic = (total_grad.array() * theta.array()).sum();

    std::cout << "Finite Diff: " << finite_diff << ", analytic: " << analytic << "\n";
    REQUIRE(fabs((analytic - finite_diff) / std::max(finite_diff, analytic)) < 1e-3);
}

TEST_CASE("neohookean_homo", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "",
                "n_refs": 0,
                "surface_selection": {
                    "threshold": 1e-6
                }
            }
        ],
        "space": {
            "discr_order": 1,
            "advanced": {
                "quadrature_order": 5
            }
        },
        "solver": {
            "nonlinear": {
                "max_step_size": 10000,
                "grad_norm": 1e-14,
                "f_delta": 1e-16
            },
            "linear": {
                "solver": "Eigen::SimplicialLDLT"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": [true, true, true]
        },
        "materials": {
            "type": "NeoHookean",
            "E": 10,
            "nu": 0.2
        }
    })"_json;
    in_args["geometry"][0]["mesh"] = path + "/../cross2d.msh";

	State state1;
	state1.init_logger("", spdlog::level::level_enum::debug, false);
	state1.init(in_args, false);

	state1.load_mesh();
	state1.build_basis();

    Eigen::MatrixXd homogenized_tensor1, homogenized_tensor2;

    state1.nl_homogenization_scale = 1e-4;
    state1.homogenization(homogenized_tensor1);

    std::cout << "Nonlinear Homogenized Tensor:\n" << homogenized_tensor1 << "\n";

    in_args["materials"]["type"] = "LinearElasticity";
	State state2;
	state2.init_logger("", spdlog::level::level_enum::err, false);
	state2.init(in_args, false);

	state2.load_mesh();
	state2.build_basis();

    Eigen::MatrixXd tmp;
    state2.homogenization(tmp);

    const int dim = state2.mesh->dimension();
    std::vector<std::pair<int, int>> unit_disp_ids;
    if (dim == 2)
        unit_disp_ids = {{0, 0}, {1, 1}, {0, 1}};
    else
        unit_disp_ids = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
    
    homogenized_tensor2.setZero(dim*dim, dim*dim);
    for (int i = 0; i < unit_disp_ids.size(); i++)
    for (int j = 0; j < unit_disp_ids.size(); j++)
    {
        homogenized_tensor2(unit_disp_ids[i].first * dim + unit_disp_ids[i].second, unit_disp_ids[j].first * dim + unit_disp_ids[j].second) = tmp(i, j);
        homogenized_tensor2(unit_disp_ids[i].second * dim + unit_disp_ids[i].first, unit_disp_ids[j].first * dim + unit_disp_ids[j].second) = tmp(i, j);
        homogenized_tensor2(unit_disp_ids[i].first * dim + unit_disp_ids[i].second, unit_disp_ids[j].second * dim + unit_disp_ids[j].first) = tmp(i, j);
        homogenized_tensor2(unit_disp_ids[i].second * dim + unit_disp_ids[i].first, unit_disp_ids[j].second * dim + unit_disp_ids[j].first) = tmp(i, j);
    }

    std::cout << "Linear Homogenized Tensor:\n" << homogenized_tensor2 << "\n";
    
    const double err = (homogenized_tensor1 - homogenized_tensor2).norm() / homogenized_tensor2.norm();
    std::cout << "Err: " << err << "\n";
    REQUIRE(err < 1e-4);
}

// TEST_CASE("density_stokes_homo", "[homogenization]")
// {
// 	const std::string path = POLYFEM_DATA_DIR;
// 	json in_args = R"({
//         "geometry": [
//             {
//                 "mesh": "",
//                 "surface_selection": {
//                     "threshold": 1e-6
//                 }
//             }
//         ],
//         "space": {
//             "discr_order": 1,
//             "pressure_discr_order": 1,
//             "advanced": {
//                 "periodic_basis": true,
//                 "quadrature_order": 2
//             }
//         },
//         "solver": {
//             "linear": {
//                 "solver": "Eigen::SimplicialLDLT"
//             }
//         },
//         "boundary_conditions": {
//             "periodic_boundary": [true, true, true]
//         },
//         "materials": {
//             "type": "Stokes",
//             "viscosity": 1,
//             "delta2": 0.00005,
//             "use_avg_pressure": false,
//             "solid_permeability": 1e-8
//         }
//     })"_json;
//     in_args["geometry"][0]["mesh"] = path + "/../2D.msh";

// 	State state;
// 	state.init_logger("", spdlog::level::level_enum::err, false);
// 	state.init(in_args, false);

// 	state.load_mesh();

// 	state.build_basis();

//     state.assemble_rhs();

//     Eigen::MatrixXd homogenized_tensor;
//     Eigen::MatrixXd density_mat(state.bases.size(), 1);
//     Eigen::MatrixXd barycenters;
//     if (state.mesh->is_volume())
//         state.mesh->cell_barycenters(barycenters);
//     else
//         state.mesh->face_barycenters(barycenters);
//     for (int e = 0; e < state.bases.size(); e++)
//     {
//         density_mat(e) = 1 - five_cylinders_fluid1(barycenters(e, 0), barycenters(e, 1));
//     }
//     // state.density.init_multimaterial(density_mat);
//     state.assembler.update_lame_params_density(density_mat);
//     state.homogenize_weighted_stokes(homogenized_tensor);

//     std::cout << homogenized_tensor << std::endl;

//     Eigen::Matrix3d reference_tensor;
//     reference_tensor <<
//     0.0271293, 1.40512e-12, 9.04009e-15,
//     1.40508e-12,    0.027081, 1.56278e-13,
//     9.04009e-15, 1.56278e-13,   0.0535809;

//     REQUIRE((homogenized_tensor - reference_tensor).norm() / reference_tensor.norm() < 1e-6);
// }

TEST_CASE("density_stokes_homo_grad", "[homogenization]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"({
        "geometry": [
            {
                "mesh": "square5.msh",
                "n_refs": 0,
                "surface_selection": {
                    "threshold": 1e-6
                },
                "transformation": {
                    "scale": 1
                }
            }
        ],
        "space": {
            "discr_order": 2,
            "pressure_discr_order": 1,
            "advanced": {
                "periodic_basis": true,
                "quadrature_order": 5
            }
        },
        "solver": {
            "linear": {
                "solver": "Eigen::SparseLU"
            }
        },
        "boundary_conditions": {
            "periodic_boundary": [true, true, true]
        },
        "materials": {
            "type": "Stokes",
            "viscosity": 1,
            "delta2": 0,
            "use_avg_pressure": true,
            "solid_permeability": 1e-8
        }
    })"_json;
    in_args["geometry"][0]["mesh"] = path + "/../square5.msh";

	State state;
	state.init_logger("", spdlog::level::level_enum::err, false);
	state.init(in_args, false);

	state.load_mesh();

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
