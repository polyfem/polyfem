////////////////////////////////////////////////////////////////////////////////
#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>

#include <iomanip>
#include <iostream>
#include <cmath>

#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <polyfem/State.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::utils;
using namespace polyfem::autogen;
namespace
{
	std::shared_ptr<State> get_state(int dim, const std::string &material_type = "NeoHookean")
	{
		const std::string path = POLYFEM_DATA_DIR;

		json material;
        material = R"(
        {
            "type": "NeoHookean",
            "E": 20000,
            "nu": 0.3,
            "rho": 1000,
            "phi": 1,
            "psi": 1
        }
        )"_json;

		json in_args = R"(
		{
			"time": {
				"dt": 0.001,
				"tend": 1.0
			},

			"output": {
				"log": {
					"level": "warning"
				}
			}

		})"_json;
		in_args["materials"] = material;
		if (dim == 2)
		{
			in_args["geometry"] = R"([{
				"enabled": true,
				"surface_selection": 7
			}])"_json;
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";
			in_args["boundary_conditions"] = R"({
				"dirichlet_boundary": [{
					"id": "all",
					"value": [0, 0]
				}],
				"rhs": [10, 10]
			})"_json;
		}
		else
		{
			in_args["geometry"] = R"([{
				"transformation": {
					"scale": [0.1, 1, 1]
				},
				"surface_selection": [
					{
						"id": 1,
						"axis": "z",
						"position": 0.8,
						"relative": true
					},
					{
						"id": 2,
						"axis": "-z",
						"position": 0.2,
						"relative": true
					},
					{
						"id": 3,
						"box": [[0, 0, 0.2], [1, 1, 0.8]],
						"relative": true
					}
				],
				"n_refs": 1
			}])"_json;
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/3D/simple/bar/bar-6.msh";
			in_args["boundary_conditions"] = R"({
				"neumann_boundary": [{
					"id": 1,
					"value": [1000, 1000, 1000]
				}],
				"pressure_boundary": [{
					"id": 1,
					"value": -2000
				},
				{
					"id": 2,
					"value": -2000
				},
				{
					"id": 3,
					"value": -2000
				}],
				"rhs": [0, 0, 0]
			})"_json;
		}

		auto state = std::make_shared<State>();
		state->init(in_args, true);
		state->set_max_threads(1);

		state->load_mesh();

		state->build_basis();
		state->assemble_rhs();
		state->assemble_mass_mat();

		return state;
	}

    template <int N>
    constexpr Eigen::Matrix<double, ((N+1)*(N+2))/2, 3> upsample_triangle()
    {
        constexpr int num = ((N+1)*(N+2))/2;

        Eigen::Matrix<double, num, 3> out;
        for (int i = 0, k = 0; i <= N; i++)
            for (int j = 0; i + j <= N; j++, k++)
            {
                std::array<int, 3> arr = {{i, j, N - i - j}};
                std::sort(arr.begin(), arr.end());

                out.row(k) << i, j, N - i - j;
            }
        
        out.template leftCols<3>() /= N;
        return out;
    }

    template <int N>
    constexpr Eigen::Matrix<double, ((N+1)*(N+2)*(N+3))/6, 4> upsample_tetrahedron()
    {
        constexpr int num = ((N+1)*(N+2)*(N+3))/6;

        Eigen::Matrix<double, num, 4> out;
        for (int i = 0, k = 0; i <= N; i++)
            for (int j = 0; j <= N; j++)
                for (int l = 0; i + j + l <= N; l++, k++)
                {
                    std::array<int, 4> arr = {{i, j, l, N - i - j - l}};
                    std::sort(arr.begin(), arr.end());

                    out.row(k) << i, j, l, N - i - j - l;
                }
        
        out.template leftCols<4>() /= N;
        return out;
    }
} // namespace

TEST_CASE("jacobian-evaluate", "[jacobian]")
{
    const double tol = 1e-8;
    constexpr int N = 7;
    for (int dim = 2; dim <= 3; dim++)
    {
        for (int order = 1; order < 4; order++)
        {
			// std::cout << "order " << order << ", dim " << dim << std::endl;
            Eigen::MatrixXd cp;
            if (dim == 2)
                autogen::p_nodes_2d(order, cp);
            else
                autogen::p_nodes_3d(order, cp);
            cp += Eigen::MatrixXd::Random(cp.rows(), cp.cols()) * 0.2;

            Eigen::MatrixXd uv;
            if (dim == 2)
                uv = upsample_triangle<N>().leftCols<2>();
            else
                uv = upsample_tetrahedron<N>().leftCols<3>();
            
            Eigen::VectorXd jac1 = robust_evaluate_jacobian(order, cp, uv);

            std::vector<Eigen::MatrixXd> grads(cp.rows(), Eigen::MatrixXd::Zero(uv.rows(), dim));
            for (int bid = 0; bid < cp.rows(); bid++)
                if (dim == 2)
                    p_grad_basis_value_2d(order, bid, uv, grads[bid]);
                else
                    p_grad_basis_value_3d(order, bid, uv, grads[bid]);
            
            Eigen::VectorXd jac2 = jac1;
            for (int k = 0; k < uv.rows(); k++)
            {
                Eigen::MatrixXd jac_mat;
                jac_mat.setZero(dim, dim);
                for (int bid = 0; bid < cp.rows(); bid++)
                    jac_mat += cp.row(bid).transpose() * grads[bid].row(k);
                
                jac2(k) = jac_mat.determinant();

				// std::cout << std::setprecision(12) << jac1(k) << ", " << jac2(k) << ", " << abs(jac1(k) - jac2(k)) / abs(jac2(k)) << std::endl;
            }

            Eigen::VectorXd denominator = jac1.array().abs().cwiseMax(jac2.array().abs()).cwiseMax(tol);
            REQUIRE(((jac2 - jac1).array() / denominator.array()).abs().maxCoeff() / tol < 1);
        }
    }
}
