#include <polyfem/State.hpp>

#include <catch2/catch.hpp>
#include <iostream>

using namespace polyfem;
using namespace polyfem::problem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;
using namespace polyfem::utils;

TEST_CASE("hessian_lin", "[assembler]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = json({});
	in_args["geometry"] = {};
	in_args["geometry"]["mesh"] = path + "/circle2.msh";
	in_args["force_linear_geometry"] = true;

	in_args["preset_problem"] = {};
	in_args["preset_problem"]["name"] = "ElasticExact";

	in_args["materials"] = {};
	in_args["materials"]["type"] = "LinearElasticity";

	State state;
	state.init_logger("", 6, false);
	state.init(in_args);
	state.load_mesh();

	state.compute_mesh_stats();
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
			state.bases, state.bases, state.ass_vals_cache, disp, mat_cache, hessian);

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
