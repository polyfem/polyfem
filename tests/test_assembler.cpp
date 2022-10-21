#include <polyfem/State.hpp>

#include <catch2/catch.hpp>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include <finitediff.hpp>

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;
using namespace polyfem::utils;

TEST_CASE("hessian_lin", "[assembler]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = json({});
	in_args["geometry"] = {};
	in_args["geometry"]["mesh"] = path + "/plane_hole.obj";
	in_args["geometry"]["surface_selection"] = 7;
	// in_args["geometry"]["mesh"] = path + "/circle2.msh";
	// in_args["force_linear_geometry"] = true;

	in_args["preset_problem"] = {};
	in_args["preset_problem"]["type"] = "ElasticExact";

	in_args["materials"] = {};
	in_args["materials"]["type"] = "LinearElasticity";
	in_args["materials"]["E"] = 1e5;
	in_args["materials"]["nu"] = 0.3;

	State state;
	state.init_logger("", spdlog::level::err, false);
	state.init(in_args, true);
	state.load_mesh();

	// state.compute_mesh_stats();
	state.build_basis();

	state.assemble_stiffness_mat();

	SpareMatrixCache mat_cache;
	StiffnessMatrix hessian;
	Eigen::MatrixXd disp(state.n_bases * 2, 1);
	disp.setZero();

	for (int rand = 0; rand < 10; ++rand)
	{
		state.assembler.assemble_energy_hessian(
			"LinearElasticity", false, state.n_bases, false,
			state.bases, state.bases, state.ass_vals_cache, 0, disp, Eigen::MatrixXd(), mat_cache, hessian);

		const StiffnessMatrix tmp = state.stiffness - hessian;
		const auto val = Approx(0).margin(1e-8);

		for (int k = 0; k < tmp.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(tmp, k); it; ++it)
			{
				REQUIRE(it.value() == val);
			}
		}

		disp.setRandom();
	}
}

double myrand(const double range = 1)
{
	return rand() / (double)RAND_MAX * range;
}

Eigen::VectorXd transform(const Eigen::VectorXd &p)
{
	Eigen::Matrix2d A, C;
	C = std::pow(1./2, 1./2) * Eigen::MatrixXd{{1, 0}, {0, -1}};
	double rand1 = myrand(0.02) + 1;
	double rand2 = myrand(0.5);
	A = std::pow(rand1, 1./3) * (rand2 * C).exp();

	double rand3 = myrand(180);
	Eigen::Matrix2d B;
	B << std::cos(rand3), -std::sin(rand3),
		 std::sin(rand3), std::cos(rand3);

	return B*A*p;
}

bool compare_matrix(
    const Eigen::MatrixXd& x,
    const Eigen::MatrixXd& y,
    const double test_eps = 1e-4)
{
    assert(x.rows() == y.rows());

    bool same = true;
	double scale = std::max(x.norm(), y.norm());
	double error = (x - y).norm();
    
	std::cout << "error: " << error << " scale: " << scale << "\n";

	if (error > scale * test_eps)
		same = false;

    return same;
}

TEST_CASE("multiscale_derivatives", "[assembler]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
	{
		"geometry": [
			{
				"mesh": "",
				"transformation": {
					"scale": 1
				},
				"volume_selection": 1,
				"surface_selection": {
					"threshold": 1e-7
				}
			}
		],
		"solver": {
			"linear": {
				"solver": "Eigen::SimplicialLDLT"
			}
		},
		"boundary_conditions": {
			"dirichlet_boundary": [
				{
					"id": 1,
					"value": [0, 0]
				}
			]
		},
		"materials": {
			"type": "MultiscaleRB",
			"microstructure": 
			{
				"geometry": [
					{
						"mesh": "",
						"n_refs": 0,
						"transformation": {
							"scale": 1e-3
						},
						"surface_selection": {
							"threshold": 1e-8
						}
					}
				],
				"solver": {
					"linear": {
						"solver": "Eigen::SimplicialLDLT"
					}
				},
				"boundary_conditions": {
					"periodic_boundary": [true, true]
				},
				"materials": {
					"type": "NeoHookean",
					"E": 100,
					"nu": 0.5
				}
			},
			"det_samples": [1, 1.1, 1.2],
			"amp_samples": [0.05, 0.15],
			"n_dir_samples": 3,
			"n_reduced_basis": 5,
			"rho": 1,
			"save_reduced_basis": "",
			"load_reduced_basis": ""
		}
	}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "../square.msh";
	in_args["materials"]["microstructure"]["geometry"][0]["mesh"] = path + "../micro30.msh";
	// in_args["materials"]["microstructure"]["materials"]["E"] = path + "../Es.txt";
	// in_args["materials"]["microstructure"]["materials"]["nu"] = path + "../nus.txt";

	State state;
	state.init_logger("", spdlog::level::warn, false);
	state.init(in_args, false);
	state.load_mesh();
	state.build_basis();

	Eigen::MatrixXd grad;
	Eigen::MatrixXd disp(state.n_bases * 2, 1);
	disp.setZero();

	utils::SpareMatrixCache mat_cache;

	for (int rand = 0; rand < 2; ++rand)
	{
		for (int p = 0; p < state.n_bases; p++)
		{
			RowVectorNd point = state.mesh_nodes->node_position(p);
			disp.block(p * 2, 0, 2, 1) = transform(point.transpose()) - point.transpose();
		}

		state.assembler.assemble_energy_gradient(
			state.formulation(), false, state.n_bases, state.bases, state.geom_bases(), 
			state.ass_vals_cache, 0, disp, disp, grad);

		Eigen::VectorXd fgrad;
		fd::finite_gradient(
			disp, [&state](const Eigen::VectorXd &x) -> double { return state.assembler.assemble_energy(state.formulation(), false, state.bases, state.geom_bases(), state.ass_vals_cache, 0, x, x); }, fgrad);

		REQUIRE (compare_matrix(grad, fgrad));

		StiffnessMatrix hessian;
		state.assembler.assemble_energy_hessian(state.formulation(), false, state.n_bases, false, state.bases, state.geom_bases(), state.ass_vals_cache, 0, disp, disp, mat_cache, hessian);
		Eigen::MatrixXd hess = hessian;

		Eigen::MatrixXd fhess;
		fd::finite_jacobian(
			disp,
			[&state](const Eigen::VectorXd &x) -> Eigen::VectorXd {
				Eigen::MatrixXd grad;
				state.assembler.assemble_energy_gradient(state.formulation(), false, state.n_bases, state.bases, state.geom_bases(), state.ass_vals_cache, 0, x, x, grad);
				return grad;
			},
			fhess);

		REQUIRE (compare_matrix(hess, fhess));
	}
}