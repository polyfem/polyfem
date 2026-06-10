#include <polyfem/State.hpp>

#include <polyfem/Units.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/assembler/NeoHookeanElasticityAutodiff.hpp>
#include <polyfem/utils/RefElementSampler.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include "VarFormTestAccess.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>

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
	state.init_logger("", spdlog::level::err, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh();

	// state.compute_mesh_stats();
	test::VarFormTestAccess::prepare(*state.variational_formulation);

	SparseMatrixCache mat_cache;
	StiffnessMatrix hessian, stiffness;
	varform::VarForm &form = *state.variational_formulation;
	const test::VarFormDebugData debug = test::VarFormTestAccess::debug_data(form);
	REQUIRE(debug.assembler != nullptr);
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.bases != nullptr);
	REQUIRE(debug.geometry_bases != nullptr);
	AssemblyValsCache ass_vals_cache;
	ass_vals_cache.init_empty();
	Eigen::MatrixXd disp(debug.n_bases * debug.mesh->dimension(), 1);
	disp.setZero();

	REQUIRE(test::VarFormTestAccess::build_stiffness_mat(form, stiffness));

	for (int rand = 0; rand < 10; ++rand)
	{
		debug.assembler->assemble_hessian(
			debug.mesh->is_volume(), debug.n_bases, false,
			*debug.bases, *debug.geometry_bases, ass_vals_cache, 0, 0, disp, Eigen::MatrixXd(), mat_cache, hessian);

		const StiffnessMatrix tmp = stiffness - hessian;
		const auto val = Catch::Approx(0).margin(1e-8);

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

TEST_CASE("hessian_hooke", "[assembler]")
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
	in_args["materials"]["type"] = "HookeLinearElasticity";
	in_args["materials"]["E"] = 1e5;
	in_args["materials"]["nu"] = 0.3;

	State state;
	state.init_logger("", spdlog::level::err, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh();

	// state.compute_mesh_stats();
	test::VarFormTestAccess::prepare(*state.variational_formulation);

	SparseMatrixCache mat_cache;
	StiffnessMatrix hessian, stiffness;
	varform::VarForm &form = *state.variational_formulation;
	const test::VarFormDebugData debug = test::VarFormTestAccess::debug_data(form);
	REQUIRE(debug.assembler != nullptr);
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.bases != nullptr);
	REQUIRE(debug.geometry_bases != nullptr);
	AssemblyValsCache ass_vals_cache;
	ass_vals_cache.init_empty();
	Eigen::MatrixXd disp(debug.n_bases * debug.mesh->dimension(), 1);
	disp.setZero();

	REQUIRE(test::VarFormTestAccess::build_stiffness_mat(form, stiffness));

	for (int rand = 0; rand < 10; ++rand)
	{
		debug.assembler->assemble_hessian(
			debug.mesh->is_volume(), debug.n_bases, false,
			*debug.bases, *debug.geometry_bases, ass_vals_cache, 0, 0, disp, Eigen::MatrixXd(), mat_cache, hessian);

		const StiffnessMatrix tmp = stiffness - hessian;
		const auto val = Catch::Approx(0).margin(1e-8);

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

TEST_CASE("generic_elastic_assembler", "[assembler]")
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
	state.init_logger("", spdlog::level::err, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh();

	// state.compute_mesh_stats();
	test::VarFormTestAccess::prepare(*state.variational_formulation);

	NeoHookeanAutodiff autodiff;
	NeoHookeanElasticity real;

	autodiff.set_size(2);
	real.set_size(2);

	Units units;
	units.init(state.args["units"]);
	const test::VarFormDebugData debug = test::VarFormTestAccess::debug_data(*state.variational_formulation);
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.bases != nullptr);
	REQUIRE(debug.geometry_bases != nullptr);

	autodiff.add_multimaterial(0, in_args["materials"], units, debug.root_path);
	real.add_multimaterial(0, in_args["materials"], units, debug.root_path);

	const int el_id = 0;
	const auto &bs = (*debug.bases)[el_id];
	const auto &gbs = (*debug.geometry_bases)[el_id];
	Eigen::MatrixXd local_pts;
	Eigen::MatrixXi f;
	regular_2d_grid(10, true, local_pts, f);

	Eigen::MatrixXd displacement(debug.n_bases, 1);

	ElementAssemblyValues vals;
	vals.compute(el_id, debug.mesh->is_volume(), bs, gbs);

	const auto &quadrature = vals.quadrature;
	const QuadratureVector da = vals.det.array() * quadrature.weights.array();

	for (int rand = 0; rand < 10; ++rand)
	{
		displacement.setRandom();

		// value
		{
			const NonLinearAssemblerData data(vals, 0, 0, displacement, displacement, da);

			const double ea = autodiff.compute_energy(data);
			const double e = real.compute_energy(data);

			if (std::isnan(e))
				REQUIRE(std::isnan(ea));
			else
				REQUIRE(ea == Catch::Approx(e).margin(1e-12));
		}

		// grad
		{
			const NonLinearAssemblerData data(vals, 0, 0, displacement, displacement, da);

			const Eigen::VectorXd grada = autodiff.assemble_gradient(data);
			const Eigen::VectorXd grad = real.assemble_gradient(data);

			for (int i = 0; i < grada.size(); ++i)
			{
				if (std::isnan(grad(i)))
					REQUIRE(std::isnan(grada(i)));
				else
					REQUIRE(grada(i) == Catch::Approx(grad(i)).margin(1e-12));
			}
		}

		// hessian
		{
			const NonLinearAssemblerData data(vals, 0, 0, displacement, displacement, da);

			const Eigen::MatrixXd hessiana = autodiff.assemble_hessian(data);
			const Eigen::MatrixXd hessian = real.assemble_hessian(data);

			for (int i = 0; i < hessiana.size(); ++i)
			{
				if (std::isnan(hessian(i)))
					REQUIRE(std::isnan(hessiana(i)));
				else
					REQUIRE(hessiana(i) == Catch::Approx(hessian(i)).margin(1e-12));
			}
		}

		// F stress
		{
			Eigen::MatrixXd stressa, stress;
			autodiff.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::F, stressa);
			real.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::F, stress);

			for (int i = 0; i < stressa.size(); ++i)
			{
				if (std::isnan(stress(i)))
					REQUIRE(std::isnan(stressa(i)));
				else
					REQUIRE(stressa(i) == Catch::Approx(stress(i)).margin(1e-12));
			}
		}

		// cauchy stress
		{
			Eigen::MatrixXd stressa, stress;
			autodiff.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::CAUCHY, stressa);
			real.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::CAUCHY, stress);

			for (int i = 0; i < stressa.size(); ++i)
			{
				if (std::isnan(stress(i)))
					REQUIRE(std::isnan(stressa(i)));
				else
					REQUIRE(stressa(i) == Catch::Approx(stress(i)).margin(1e-12));
			}
		}

		// pk1 stress
		{
			Eigen::MatrixXd stressa, stress;
			autodiff.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::PK1, stressa);
			real.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::PK1, stress);

			for (int i = 0; i < stressa.size(); ++i)
			{
				if (std::isnan(stress(i)))
					REQUIRE(std::isnan(stressa(i)));
				else
					REQUIRE(stressa(i) == Catch::Approx(stress(i)).margin(1e-12));
			}
		}

		// pk2 stress
		{
			Eigen::MatrixXd stressa, stress;
			autodiff.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::PK2, stressa);
			real.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::PK2, stress);

			for (int i = 0; i < stressa.size(); ++i)
			{
				if (std::isnan(stress(i)))
					REQUIRE(std::isnan(stressa(i)));
				else
					REQUIRE(stressa(i) == Catch::Approx(stress(i)).margin(1e-12));
			}
		}
	}
}
