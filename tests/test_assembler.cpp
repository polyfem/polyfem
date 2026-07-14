#include <polyfem/State.hpp>

#include <polyfem/Units.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/assembler/NeoHookeanElasticityAutodiff.hpp>
#include <polyfem/assembler/VolumePenalty.hpp>
#include <polyfem/utils/RefElementSampler.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include "VarFormTestAccess.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <algorithm>

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;
using namespace polyfem::utils;

namespace
{
	class RecordingAssembler : public Assembler
	{
	public:
		std::string name() const override { return "RecordingAssembler"; }
		std::map<std::string, ParamFunc> parameters() const override { return {}; }
		bool is_linear() const override { return true; }

		void add_multimaterial(const int index, const json &params, const Units &, const std::string &root_path) override
		{
			indices.push_back(index);
			materials.push_back(params);
			root_paths.push_back(root_path);
		}

		std::vector<int> indices;
		std::vector<json> materials;
		std::vector<std::string> root_paths;
	};

	class ConfigurableNeoHookeanAutodiff : public NeoHookeanAutodiff
	{
	public:
		void set_autodiff_type(const AutodiffType type) { autodiff_type_ = type; }
	};

	class ConfigurableVolumePenalty : public VolumePenalty
	{
	public:
		void set_autodiff_type(const AutodiffType type) { autodiff_type_ = type; }
	};

	struct SyntheticNonlinearElement
	{
		ElementAssemblyValues vals;
		Eigen::MatrixXd x;
		Eigen::MatrixXd x_prev;
		QuadratureVector da;
	};

	SyntheticNonlinearElement make_synthetic_nonlinear_element(const int dim, const int n_bases)
	{
		SyntheticNonlinearElement result;
		result.vals.element_id = 0;
		result.vals.is_volume_ = dim == 3;
		result.vals.basis_values.resize(n_bases);
		result.vals.val.resize(1, dim);
		for (int d = 0; d < dim; ++d)
			result.vals.val(0, d) = 0.1 * (d + 1);
		result.vals.quadrature.points = result.vals.val;
		result.vals.quadrature.weights = Eigen::VectorXd::Ones(1);
		result.vals.det = Eigen::VectorXd::Ones(1);
		result.vals.jac_it = {Eigen::MatrixXd::Identity(dim, dim)};

		for (int i = 0; i < n_bases; ++i)
		{
			AssemblyValues &basis = result.vals.basis_values[i];
			RowVectorNd node(dim);
			for (int d = 0; d < dim; ++d)
				node(d) = 0.01 * (i + 1) * (d + 1);
			basis.global = {Local2Global(i, node, 1.0)};
			basis.val = Eigen::MatrixXd::Constant(1, 1, 1.0 / n_bases);
			basis.grad.resize(1, dim);
			for (int d = 0; d < dim; ++d)
				basis.grad(0, d) = 0.015 * (i + 1) * (d + 1) / n_bases;
			basis.grad_t_m = basis.grad;
		}

		result.x.resize(n_bases * dim, 1);
		result.x_prev.resize(n_bases * dim, 1);
		for (int i = 0; i < result.x.rows(); ++i)
		{
			result.x(i) = 0.002 * (i + 1);
			result.x_prev(i) = -0.001 * (i + 1);
		}
		result.da = QuadratureVector::Constant(1, 1.25);
		return result;
	}

	void require_approx_vector(const Eigen::VectorXd &actual, const Eigen::VectorXd &expected, const double margin = 1e-8)
	{
		REQUIRE(actual.size() == expected.size());
		for (int i = 0; i < actual.size(); ++i)
			REQUIRE(actual(i) == Catch::Approx(expected(i)).margin(margin));
	}

	void require_approx_matrix(const Eigen::MatrixXd &actual, const Eigen::MatrixXd &expected, const double margin = 1e-8)
	{
		REQUIRE(actual.rows() == expected.rows());
		REQUIRE(actual.cols() == expected.cols());
		for (int i = 0; i < actual.size(); ++i)
			REQUIRE(actual(i) == Catch::Approx(expected(i)).margin(margin));
	}
} // namespace

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

TEST_CASE("neo hookean synthetic nonlinear branches", "[assembler][elasticity]")
{
	const std::vector<std::pair<int, int>> cases = {
		{2, 3}, {2, 6}, {2, 10}, {2, 5}, {3, 4}, {3, 10}, {3, 20}, {3, 5}};

	for (const auto &[dim, n_bases] : cases)
	{
		CAPTURE(dim);
		CAPTURE(n_bases);

		Units units;
		NeoHookeanElasticity fast;
		ConfigurableNeoHookeanAutodiff full;
		ConfigurableNeoHookeanAutodiff stress;
		fast.set_size(dim);
		full.set_size(dim);
		stress.set_size(dim);
		full.set_autodiff_type(AutodiffType::FULL);
		stress.set_autodiff_type(AutodiffType::STRESS);

		const json material = {{"E", 12.0}, {"nu", 0.23}};
		fast.add_multimaterial(0, material, units, "");
		full.add_multimaterial(0, material, units, "");
		stress.add_multimaterial(0, material, units, "");

		const SyntheticNonlinearElement fixture = make_synthetic_nonlinear_element(dim, n_bases);
		const NonLinearAssemblerData data(fixture.vals, 0.2, 0.01, fixture.x, fixture.x_prev, fixture.da);

		REQUIRE(fast.compute_energy(data) == Catch::Approx(full.compute_energy(data)).margin(1e-10));
		require_approx_vector(fast.assemble_gradient(data), full.assemble_gradient(data), 1e-8);
		require_approx_vector(stress.assemble_gradient(data), full.assemble_gradient(data), 1e-8);
		require_approx_matrix(fast.assemble_hessian(data), full.assemble_hessian(data), 1e-7);
		require_approx_matrix(stress.assemble_hessian(data), full.assemble_hessian(data), 1e-7);
	}
}

TEST_CASE("generic elastic autodiff modes and stress products", "[assembler][elasticity]")
{
	const std::vector<std::pair<int, int>> cases = {
		{2, 3}, {2, 6}, {2, 10}, {2, 5}, {3, 4}, {3, 10}, {3, 20}, {3, 5}};

	for (const auto &[dim, n_bases] : cases)
	{
		CAPTURE(dim);
		CAPTURE(n_bases);

		Units units;
		ConfigurableVolumePenalty full;
		ConfigurableVolumePenalty stress;
		ConfigurableVolumePenalty no_ad;
		full.set_size(dim);
		stress.set_size(dim);
		no_ad.set_size(dim);
		full.set_autodiff_type(AutodiffType::FULL);
		stress.set_autodiff_type(AutodiffType::STRESS);
		no_ad.set_autodiff_type(AutodiffType::NONE);

		const json material = {{"k", 3.0}};
		full.add_multimaterial(0, material, units, "");
		stress.add_multimaterial(0, material, units, "");
		no_ad.add_multimaterial(0, material, units, "");

		const SyntheticNonlinearElement fixture = make_synthetic_nonlinear_element(dim, n_bases);
		const NonLinearAssemblerData data(fixture.vals, 0.2, 0.01, fixture.x, fixture.x_prev, fixture.da);

		REQUIRE(stress.compute_energy(data) == Catch::Approx(full.compute_energy(data)).margin(1e-12));
		REQUIRE(no_ad.compute_energy(data) == Catch::Approx(full.compute_energy(data)).margin(1e-12));
		require_approx_vector(stress.assemble_gradient(data), full.assemble_gradient(data), 1e-9);
		require_approx_vector(no_ad.assemble_gradient(data), full.assemble_gradient(data), 1e-9);
		require_approx_matrix(stress.assemble_hessian(data), full.assemble_hessian(data), 1e-8);
		require_approx_matrix(no_ad.assemble_hessian(data), full.assemble_hessian(data), 1e-8);
	}

	ConfigurableVolumePenalty assembler;
	assembler.set_size(3);
	assembler.set_autodiff_type(AutodiffType::STRESS);
	Units units;
	assembler.add_multimaterial(0, {{"k", 3.0}}, units, "");

	Eigen::MatrixXd grad_u(3, 3);
	grad_u << 0.03, 0.01, 0.00,
		-0.02, 0.02, 0.01,
		0.00, -0.01, 0.04;
	Eigen::MatrixXd local_pt(1, 3);
	local_pt << 0.1, 0.2, 0.3;
	Eigen::MatrixXd global_pt(1, 3);
	global_pt << 0.4, 0.5, 0.6;
	const OptAssemblerData opt_data(0.2, 0.01, 0, local_pt, global_pt, grad_u);

	Eigen::MatrixXd stress;
	Eigen::MatrixXd mat_result;
	assembler.compute_stress_grad_multiply_mat(opt_data, Eigen::Matrix3d::Identity(), stress, mat_result);
	REQUIRE(stress.rows() == 3);
	REQUIRE(stress.cols() == 3);
	REQUIRE(mat_result.rows() == 3);
	REQUIRE(mat_result.cols() == 3);

	Eigen::MatrixXd stress_again;
	Eigen::MatrixXd stress_result;
	assembler.compute_stress_grad_multiply_stress(opt_data, stress_again, stress_result);
	require_approx_matrix(stress_again, stress, 1e-12);

	Eigen::MatrixXd expected_stress_result_stress;
	Eigen::MatrixXd expected_stress_result;
	assembler.compute_stress_grad_multiply_mat(opt_data, stress, expected_stress_result_stress, expected_stress_result);
	require_approx_matrix(expected_stress_result_stress, stress, 1e-12);
	require_approx_matrix(stress_result, expected_stress_result, 1e-12);

	const Eigen::RowVector3d row_vect(1, 2, 3);
	Eigen::MatrixXd row_stress;
	Eigen::MatrixXd row_result;
	assembler.compute_stress_grad_multiply_vect(opt_data, row_vect, row_stress, row_result);
	REQUIRE(row_result.rows() == 9);
	REQUIRE(row_result.cols() == 3);
	require_approx_matrix(row_stress, stress, 1e-12);

	const Eigen::Vector3d col_vect(1, -1, 2);
	Eigen::MatrixXd col_stress;
	Eigen::MatrixXd col_result;
	assembler.compute_stress_grad_multiply_vect(opt_data, col_vect, col_stress, col_result);
	REQUIRE(col_result.rows() == 9);
	REQUIRE(col_result.cols() == 3);
	require_approx_matrix(col_stress, stress, 1e-12);
}

TEST_CASE("material parameter helpers", "[assembler][mat_params]")
{
	SECTION("generic scalar parameters")
	{
		GenericMatParam parameter("coefficient");
		parameter.add_multimaterial(0, {{"coefficient", 2.5}}, "", "");
		parameter.add_multimaterial(1, {{"coefficient", "x + 2*y + t"}}, "", "");

		RowVectorNd point(2);
		point << 3, 4;
		REQUIRE(parameter(point, 5, 0) == Catch::Approx(2.5));
		REQUIRE(parameter(point, 5, 1) == Catch::Approx(16));
		REQUIRE(parameter(3, 4, 7, 5, 1) == Catch::Approx(16));
	}

	SECTION("generic parameter arrays")
	{
		GenericMatParams parameters("coefficients");
		parameters.add_multimaterial(0, {{"unused", 1}}, "", "");
		REQUIRE(parameters.size() == 0);

		parameters.add_multimaterial(0, {{"coefficients", json::array({1.0, "x + y"})}}, "", "");
		parameters.add_multimaterial(1, {{"coefficients", json::array({3.0, 4.0})}}, "", "");
		REQUIRE(parameters.size() == 2);
		REQUIRE(parameters[0](0, 0, 0, 0, 0) == Catch::Approx(1));
		REQUIRE(parameters[1](2, 5, 0, 0, 0) == Catch::Approx(7));
		REQUIRE(parameters[0](0, 0, 0, 0, 1) == Catch::Approx(3));
		REQUIRE(parameters[1](0, 0, 0, 0, 1) == Catch::Approx(4));
	}

	SECTION("elasticity tensors")
	{
		ElasticityTensor tensor;
		tensor.resize(2);
		tensor.set_from_lambda_mu(2, 3, "", "");
		REQUIRE(tensor(0, 0) == Catch::Approx(8));
		REQUIRE(tensor(1, 0) == Catch::Approx(2));
		REQUIRE(tensor(2, 2) == Catch::Approx(3));

		const std::array<double, 3> strain = {{1, 2, 3}};
		REQUIRE(tensor.compute_stress<3>(strain, 0) == Catch::Approx(12));

		tensor.set_from_entries({10, 2, 0, 20, 0, 5}, "", "");
		REQUIRE(tensor(0, 0) == Catch::Approx(10));
		REQUIRE(tensor(0, 1) == Catch::Approx(2));
		REQUIRE(tensor(2, 2) == Catch::Approx(5));

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6> rotation(3, 3);
		rotation << 0, 1, 0, 1, 0, 0, 0, 0, 1;
		tensor.rotate_stiffness(rotation);
		REQUIRE(tensor(0, 0) == Catch::Approx(20));
		tensor.unrotate_stiffness();
		REQUIRE(tensor(0, 0) == Catch::Approx(10));

		tensor.set_from_young_poisson(100, 0.25, "", "");
		REQUIRE(tensor(0, 0) == Catch::Approx(100.0 / (1 - 0.25 * 0.25)));

		ElasticityTensor tensor3d;
		tensor3d.resize(3);
		tensor3d.set_from_lambda_mu(2, 3, "", "");
		REQUIRE(tensor3d(0, 0) == Catch::Approx(8));
		REQUIRE(tensor3d(0, 2) == Catch::Approx(2));
		REQUIRE(tensor3d(5, 5) == Catch::Approx(3));
	}

	SECTION("fiber directions")
	{
		FiberDirection identity;
		identity.resize(3);
		identity.add_multimaterial(0, json::array(), "", "");
		REQUIRE_FALSE(identity.has_rotation());
		REQUIRE(identity(0, 0, 0, 0, 0, 0, 0, 0).isApprox(Eigen::Matrix3d::Identity()));

		FiberDirection fibers;
		fibers.resize(3);
		fibers.add_multimaterial(0, json::array({0, 1, 0, 1, 0, 0, 0, 0, 1}), "", "");
		REQUIRE(fibers.has_rotation());
		const auto direction = fibers(0, 0, 0, 0, 0, 0, 0, 0);
		REQUIRE(direction(0, 1) == Catch::Approx(1));
		REQUIRE(direction(1, 0) == Catch::Approx(1));
		const auto voigt = fibers.stiffness_rotation_voigt(0, 0, 0, 0, 0, 0, 0, 0);
		REQUIRE(voigt.rows() == 6);
		REQUIRE(voigt.cols() == 6);
		REQUIRE(voigt(0, 1) == Catch::Approx(1));
		REQUIRE(voigt(1, 0) == Catch::Approx(1));
	}
}

TEST_CASE("density and Lame parameter helpers", "[assembler][mat_params]")
{
	SECTION("Lame parameters")
	{
		LameParameters plane_stress;
		plane_stress.add_multimaterial(0, {{"E", 100.0}, {"nu", 0.25}}, false, "", "");
		double lambda = 0;
		double mu = 0;
		plane_stress.lambda_mu(0, 0, 0, 0, 0, 0, 0, 0, lambda, mu);
		REQUIRE(lambda == Catch::Approx(100.0 * 0.25 / (1 - 0.25 * 0.25)));
		REQUIRE(mu == Catch::Approx(40));

		LameParameters volume;
		volume.add_multimaterial(0, {{"young", 100.0}, {"nu", 0.25}}, true, "", "");
		volume.lambda_mu(0, 0, 0, 0, 0, 0, 0, 0, lambda, mu);
		REQUIRE(lambda == Catch::Approx(40));
		REQUIRE(mu == Catch::Approx(40));

		LameParameters direct;
		direct.add_multimaterial(0, {{"lambda", 7.0}, {"mu", 11.0}}, true, "", "");
		direct.lambda_mu(0, 0, 0, 0, 0, 0, 0, 0, lambda, mu);
		REQUIRE(lambda == Catch::Approx(7));
		REQUIRE(mu == Catch::Approx(11));
		direct.lambda_mat_ = Eigen::VectorXd::Constant(1, 13);
		direct.mu_mat_ = Eigen::VectorXd::Constant(1, 17);
		direct.lambda_mu(0, 0, 0, 0, 0, 0, 0, 0, lambda, mu);
		REQUIRE(lambda == Catch::Approx(13));
		REQUIRE(mu == Catch::Approx(17));
	}

	SECTION("densities")
	{
		Density density;
		density.add_multimaterial(0, {{"rho", 2.5}}, "", "");
		density.add_multimaterial(1, {{"density", "x + y"}}, "", "");
		REQUIRE(density(0, 0, 0, 0, 0, 0, 0, 0) == Catch::Approx(2.5));
		REQUIRE(density(0, 0, 0, 2, 3, 0, 0, 1) == Catch::Approx(5));

		Eigen::Vector2d param(0, 0);
		Eigen::Vector2d point(2, 3);
		REQUIRE(density(param, point, 0, 1) == Catch::Approx(5));

		NoDensity no_density;
		REQUIRE(no_density(param, point, 0, 42) == Catch::Approx(1));
		REQUIRE_THROWS(no_density.add_multimaterial(0, json::object(), "", ""));
	}

	SECTION("thermal mass density")
	{
		ThermalMassDensity density;
		density.add_multimaterial(0, {{"rho", 3.0}, {"heat_capacity", 4.0}}, "", "", "");
		REQUIRE(density(0, 0, 0, 0, 0, 0, 0, 0) == Catch::Approx(12));

		RowVectorNd point(3);
		point << 1, 2, 3;
		REQUIRE(density.rho(point, 0, 0) == Catch::Approx(3));
		REQUIRE(density.heat_capacity(point, 0, 0) == Catch::Approx(4));
	}
}

TEST_CASE("assembler utility factories and metadata", "[assembler][assembler_utils]")
{
	REQUIRE(AssemblerUtils::other_assembler_name("Bilaplacian") == "BilaplacianAux");
	REQUIRE(AssemblerUtils::other_assembler_name("Stokes") == "StokesPressure");
	REQUIRE(AssemblerUtils::other_assembler_name("NavierStokes") == "StokesPressure");
	REQUIRE(AssemblerUtils::other_assembler_name("OperatorSplitting") == "StokesPressure");
	REQUIRE(AssemblerUtils::other_assembler_name("IncompressibleLinearElasticity") == "IncompressibleLinearElasticityPressure");
	REQUIRE(AssemblerUtils::other_assembler_name("Laplacian").empty());

	const std::vector<std::string> formulations = {
		"Helmholtz", "Laplacian", "Electrostatics", "Bilaplacian", "BilaplacianAux",
		"LinearElasticity", "HookeLinearElasticity", "IncompressibleLinearElasticity",
		"IncompressibleLinearElasticityPressure", "SaintVenant", "NeoHookean",
		"IsochoricNeoHookean", "MooneyRivlin", "MooneyRivlin3Param",
		"MooneyRivlin3ParamSymbolic", "MultiModels", "MaterialSum", "UnconstrainedOgden",
		"IncompressibleOgden", "VolumePenalty", "HGOFiber", "ActiveFiber", "Stokes",
		"StokesPressure", "NavierStokes", "OperatorSplitting", "AMIPS", "FixedCorotational"};
	for (const std::string &formulation : formulations)
		REQUIRE(AssemblerUtils::make_assembler(formulation) != nullptr);
	REQUIRE_THROWS(AssemblerUtils::make_assembler("not-an-assembler"));

	REQUIRE(AssemblerUtils::make_mixed_assembler("Bilaplacian")->name() == "BilaplacianMixed");
	REQUIRE(AssemblerUtils::make_mixed_assembler("IncompressibleLinearElasticity")->name() == "IncompressibleLinearElasticityMixed");
	REQUIRE(AssemblerUtils::make_mixed_assembler("Stokes")->name() == "StokesMixed");
	REQUIRE(AssemblerUtils::make_mixed_assembler("NavierStokes")->name() == "StokesMixed");
	REQUIRE(AssemblerUtils::make_mixed_assembler("OperatorSplitting")->name() == "StokesMixed");
	REQUIRE_THROWS(AssemblerUtils::make_mixed_assembler("Laplacian"));
	REQUIRE(AssemblerUtils::make_mixed_nl_assembler("ThermoElasticity")->name() == "ThermoElasticity");
	REQUIRE_THROWS(AssemblerUtils::make_mixed_nl_assembler("Laplacian"));

	using BasisType = AssemblerUtils::BasisType;
	REQUIRE(AssemblerUtils::quadrature_order("Mass", 0, BasisType::SIMPLEX_LAGRANGE, 2) == 1);
	REQUIRE(AssemblerUtils::quadrature_order("Mass", 2, BasisType::CUBE_LAGRANGE, 3) == 4);
	REQUIRE(AssemblerUtils::quadrature_order("Mass", 2, BasisType::SPLINE, 2) == 5);
	REQUIRE(AssemblerUtils::quadrature_order("NavierStokes", 2, BasisType::SIMPLEX_LAGRANGE, 2) == 3);
	REQUIRE(AssemblerUtils::quadrature_order("NavierStokes", 2, BasisType::CUBE_LAGRANGE, 2) == 4);
	REQUIRE(AssemblerUtils::quadrature_order("NavierStokes", 2, BasisType::SPLINE, 2) == 5);
	REQUIRE(AssemblerUtils::quadrature_order("Laplacian", 2, BasisType::SIMPLEX_LAGRANGE, 2) == 2);
	REQUIRE(AssemblerUtils::quadrature_order("Laplacian", 2, BasisType::PRISM_LAGRANGE, 3) == 4);
	REQUIRE(AssemblerUtils::quadrature_order("Laplacian", 2, BasisType::POLY, 2) == 3);

	const auto elastic_materials = AssemblerUtils::elastic_materials();
	REQUIRE(std::find(elastic_materials.begin(), elastic_materials.end(), "NeoHookean") != elastic_materials.end());
	REQUIRE(AssemblerUtils::is_elastic_material("LinearElasticity"));
	REQUIRE_FALSE(AssemblerUtils::is_elastic_material("Laplacian"));
}

TEST_CASE("merge mixed assembler matrices", "[assembler][assembler_utils]")
{
	StiffnessMatrix velocity(2, 2);
	velocity.insert(0, 0) = 2;
	velocity.insert(1, 1) = 3;
	StiffnessMatrix mixed(2, 1);
	mixed.insert(0, 0) = 5;
	mixed.insert(1, 0) = 7;
	StiffnessMatrix pressure(1, 1);
	pressure.insert(0, 0) = 11;

	StiffnessMatrix result;
	AssemblerUtils::merge_mixed_matrices(1, 1, 2, false, velocity, mixed, pressure, result);
	Eigen::Matrix3d expected;
	expected << 2, 0, 5, 0, 3, 7, 5, 7, 11;
	REQUIRE(Eigen::MatrixXd(result).isApprox(expected));

	AssemblerUtils::merge_mixed_matrices(1, 1, 2, true, velocity, mixed, pressure, result);
	Eigen::Matrix4d expected_with_average = Eigen::Matrix4d::Zero();
	expected_with_average.topLeftCorner<3, 3>() = expected;
	expected_with_average(2, 3) = 1;
	expected_with_average(3, 2) = 1;
	REQUIRE(Eigen::MatrixXd(result).isApprox(expected_with_average));
}

TEST_CASE("assembler material dispatch", "[assembler]")
{
	Units units;
	RecordingAssembler assembler;

	assembler.set_materials({4, 4}, {{"value", 3}}, units, "/root/path");
	REQUIRE(assembler.indices == std::vector<int>{0});
	REQUIRE(assembler.materials[0]["value"] == 3);
	REQUIRE(assembler.root_paths[0] == "/root/path");

	assembler.indices.clear();
	assembler.materials.clear();
	assembler.root_paths.clear();
	const json materials = json::array({{{"id", json::array({7, 9})}, {"value", 70}},
										{{"id", 8}, {"value", 80}}});
	assembler.set_materials({7, 8, 9, 10}, materials, units, "/materials");
	REQUIRE(assembler.indices == std::vector<int>{0, 1, 2});
	REQUIRE(assembler.materials[0]["value"] == 70);
	REQUIRE(assembler.materials[1]["value"] == 80);
	REQUIRE(assembler.materials[2]["value"] == 70);
	REQUIRE(assembler.root_paths == std::vector<std::string>{"/materials", "/materials", "/materials"});
}
