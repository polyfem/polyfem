#include <polyfem/State.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>
#include <polyfem/solver/forms/NavierStokesFSIForm.hpp>

#include "VarFormTestAccess.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <fstream>
#include <filesystem>
#include <set>
#include <string>
#include <utility>

using namespace polyfem;

namespace
{
	json transient_args()
	{
		json args = json::object();
		args["time"] = json::object();
		return args;
	}

	json load_scene(const std::string &path)
	{
		std::ifstream file(path);
		REQUIRE(file.is_open());

		json args;
		file >> args;
		args["root_path"] = path;
		args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
		args["/output/directory"_json_pointer] = "";
		args["/output/log/quiet"_json_pointer] = true;
		args["/output/log/level"_json_pointer] = "error";
		args["/output/advanced/save_time_sequence"_json_pointer] = false;
		args["/output/paraview/file_name"_json_pointer] = "";
		args["/output/data/state"_json_pointer] = "";

		return args;
	}
} // namespace

TEST_CASE("varform factory supports migrated formulations", "[varform]")
{
	const json args = transient_args();

	for (const std::string formulation : {
			 "NeoHookean",
			 "LinearElasticity",
			 "Laplacian",
			 "Stokes",
			 "NavierStokes",
			 "OperatorSplitting",
			 "IncompressibleLinearElasticity",
			 "Bilaplacian",
		 })
	{
		CHECK(varform::VarFormFactory::supports(formulation, args));
		CHECK(varform::VarFormFactory::create(formulation, args) != nullptr);
	}
	CHECK(varform::VarFormFactory::supports("NavierStokesFSI", args));
	CHECK(varform::VarFormFactory::create("NavierStokesFSI", args) != nullptr);
	json static_args = args;
	static_args["time"] = nullptr;
	CHECK_FALSE(varform::VarFormFactory::supports("NavierStokesFSI", static_args));
	CHECK(varform::VarFormFactory::create("NavierStokesFSI", static_args) == nullptr);

	json periodic_args = args;
	periodic_args["/boundary_conditions/periodic_boundary/enabled"_json_pointer] = true;
	CHECK_FALSE(varform::VarFormFactory::supports("Stokes", periodic_args));
	CHECK(varform::VarFormFactory::create("Stokes", periodic_args) == nullptr);
	CHECK_FALSE(varform::VarFormFactory::supports("NeoHookean", periodic_args));
	CHECK(varform::VarFormFactory::create("NeoHookean", periodic_args) == nullptr);
}

TEST_CASE("state can opt into migrated varforms", "[varform][state]")
{
	for (const auto &[scene, expected_name] : {
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/stokes_static.json", std::string("Stokes")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/units/navier_stokes_static.json", std::string("NavierStokes")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/navier_stokes_split.json", std::string("OperatorSplitting")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/incompressible.json", std::string("IncompressibleElastic")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/bilaplace.json", std::string("Bilaplacian")},
		 })
	{
		State state;
		state.init(load_scene(scene), true);

		REQUIRE(state.variational_formulation != nullptr);
		CHECK(state.variational_formulation->name() == expected_name);
	}
}

TEST_CASE("ALE Navier-Stokes FSI runs through State", "[varform][state][navier_stokes][fsi]")
{
	const std::filesystem::path state_pattern =
		std::filesystem::temp_directory_path() / "polyfem-navier-stokes-fsi-state-{:d}.h5";
	const std::string state_path =
		(std::filesystem::temp_directory_path() / "polyfem-navier-stokes-fsi-state-1.h5").string();
	std::filesystem::remove(state_path);
	json args = load_scene(std::string(POLYFEM_DATA_DIR) + "/standard/navier_stokes_transient.json");
	args.erase("preset_problem");
	args["geometry"] = {
		{"mesh", std::string(POLYFEM_DATA_DIR) + "/contact/meshes/2D/simple/square.obj"},
		{"surface_selection", json::array({{{"id", 7}, {"box", {{-10, -10}, {10, 10}}}}})}};
	args["materials"] = {
		{"type", "NavierStokesFSI"},
		{"viscosity", 0.1},
		{"rho", 1.0},
		{"velocity_space_id", 0},
		{"pressure_space_id", 1},
		{"mesh_displacement_space_id", 2},
		{"mesh_material", {{"type", "NeoHookean"}, {"E", 10.0}, {"nu", 0.3}, {"rho", 1.0}}}};
	args["boundary_conditions"] = {
		{"dirichlet_boundary", json::array({{{"id", 7}, {"fe_space", 0}, {"value", {0, 0}}},
											{{"id", 7}, {"fe_space", 2}, {"value", {"0.1*t", 0}}}})}};
	args["space"]["discr_order"] = json::array({{{"fe_space", 0}, {"order", 2}},
												{{"fe_space", 1}, {"order", 1}},
												{{"fe_space", 2}, {"order", 1}}});
	args["time"] = {{"t0", 0}, {"tend", 0.01}, {"time_steps", 1}};
	args["/output/data/state"_json_pointer] = state_pattern.string();

	State state;
	state.init(args, true);
	state.load_mesh();
	Eigen::MatrixXd solution;
	state.solve(solution);

	REQUIRE(state.variational_formulation != nullptr);
	CHECK(state.variational_formulation->name() == "NavierStokesFSI");
	CHECK(solution.rows() > 0);
	CHECK(solution.allFinite());
	REQUIRE(std::filesystem::is_regular_file(state_path));

	const test::NavierStokesFSIDebugData debug =
		test::VarFormTestAccess::navier_stokes_fsi_data(*state.variational_formulation);
	REQUIRE(debug.average_pressure_form != nullptr);
	const int pressure_offset = debug.velocity_ndof;
	const int mesh_offset = pressure_offset + debug.pressure_ndof;
	const int multiplier_offset = mesh_offset + debug.mesh_displacement_ndof;
	Eigen::VectorXd solved_gauge_residual;
	debug.average_pressure_form->first_derivative(solution.col(0), solved_gauge_residual);
	CHECK(std::abs(solved_gauge_residual(multiplier_offset)) < 1e-9);
	Eigen::VectorXd gauge_x = solution.col(0);
	gauge_x.segment(pressure_offset, debug.pressure_ndof) =
		Eigen::VectorXd::LinSpaced(debug.pressure_ndof, -0.4, 0.7);
	for (int i = 0; i < debug.mesh_displacement_ndof; ++i)
		gauge_x(mesh_offset + i) = 0.01 * std::sin(double(i + 1));
	gauge_x(multiplier_offset) = 0.6;

	double pressure_integral = 0;
	double volume = 0;
	for (int e = 0; e < int(debug.geometry_bases->size()); ++e)
	{
		assembler::ElementAssemblyValues pressure_vals, displacement_vals;
		debug.pressure_cache->compute(
			e, debug.is_volume, debug.pressure_bases->at(e),
			debug.geometry_bases->at(e), pressure_vals);
		debug.mesh_displacement_cache->compute(
			e, debug.is_volume, debug.mesh_displacement_bases->at(e),
			debug.geometry_bases->at(e), displacement_vals);
		if (pressure_vals.quadrature.weights.size() != displacement_vals.quadrature.weights.size())
		{
			displacement_vals.compute(
				e, debug.is_volume, pressure_vals.quadrature.points,
				debug.mesh_displacement_bases->at(e), debug.geometry_bases->at(e));
			displacement_vals.quadrature = pressure_vals.quadrature;
		}
		Eigen::VectorXd local_displacement = Eigen::VectorXd::Zero(
			int(displacement_vals.basis_values.size()) * 2);
		Eigen::VectorXd local_pressure = Eigen::VectorXd::Zero(
			int(pressure_vals.basis_values.size()));
		for (int a = 0; a < int(displacement_vals.basis_values.size()); ++a)
			for (int c = 0; c < 2; ++c)
				for (const auto &global : displacement_vals.basis_values[a].global)
					local_displacement(a * 2 + c) += global.val * gauge_x(mesh_offset + global.index * 2 + c);
		for (int a = 0; a < int(pressure_vals.basis_values.size()); ++a)
			for (const auto &global : pressure_vals.basis_values[a].global)
				local_pressure(a) += global.val * gauge_x(pressure_offset + global.index);

		for (int q = 0; q < pressure_vals.quadrature.weights.size(); ++q)
		{
			Eigen::Matrix2d F = Eigen::Matrix2d::Identity();
			for (int a = 0; a < int(displacement_vals.basis_values.size()); ++a)
			{
				const Eigen::RowVector2d grad =
					displacement_vals.basis_values[a].grad.row(q) * displacement_vals.jac_it[q];
				for (int c = 0; c < 2; ++c)
					F.row(c) += local_displacement(a * 2 + c) * grad;
			}
			const double weight = pressure_vals.det(q)
								  * pressure_vals.quadrature.weights(q) * F.determinant();
			double pressure = 0;
			for (int a = 0; a < local_pressure.size(); ++a)
				pressure += pressure_vals.basis_values[a].val(q) * local_pressure(a);
			pressure_integral += weight * pressure;
			volume += weight;
		}
	}
	Eigen::VectorXd gauge_residual;
	debug.average_pressure_form->first_derivative(gauge_x, gauge_residual);
	CHECK(gauge_residual(multiplier_offset)
		  == Catch::Approx(debug.average_pressure_form->weight() * pressure_integral / volume).margin(1e-11));
	CHECK(gauge_residual.segment(mesh_offset, debug.mesh_displacement_ndof).isZero(0));

	Eigen::VectorXd constant_pressure_x = gauge_x;
	constant_pressure_x.segment(pressure_offset, debug.pressure_ndof).setOnes();
	constant_pressure_x(multiplier_offset) = 0;
	debug.average_pressure_form->first_derivative(constant_pressure_x, gauge_residual);
	CHECK(gauge_residual(multiplier_offset)
		  == Catch::Approx(debug.average_pressure_form->weight()).margin(1e-12));

	StiffnessMatrix gauge_jacobian;
	debug.average_pressure_form->second_derivative(gauge_x, gauge_jacobian);
	for (int j = pressure_offset; j <= multiplier_offset; ++j)
	{
		if (j >= mesh_offset + debug.mesh_displacement_ndof && j != multiplier_offset)
			continue;
		const double eps = 1e-7;
		Eigen::VectorXd plus = gauge_x;
		Eigen::VectorXd minus = gauge_x;
		plus(j) += eps;
		minus(j) -= eps;
		Eigen::VectorXd plus_residual, minus_residual;
		debug.average_pressure_form->first_derivative(plus, plus_residual);
		debug.average_pressure_form->first_derivative(minus, minus_residual);
		const Eigen::VectorXd finite_difference = (plus_residual - minus_residual) / (2 * eps);
		CHECK(Eigen::VectorXd(gauge_jacobian.col(j)).isApprox(finite_difference, 2e-7));
	}

	json restart_args = args;
	restart_args["/input/data/state"_json_pointer] = state_path;
	restart_args["/output/data/state"_json_pointer] = "";
	restart_args["time"] = {{"t0", 0.01}, {"tend", 0.02}, {"time_steps", 1}};
	State restarted_state;
	restarted_state.init(restart_args, true);
	restarted_state.load_mesh();
	Eigen::MatrixXd restarted_solution;
	restarted_state.solve(restarted_solution);
	CHECK(restarted_solution.rows() == solution.rows());
	CHECK(restarted_solution.allFinite());
	std::filesystem::remove(state_path);
}

TEST_CASE("ALE Navier-Stokes FSI moving arc", "[varform][state][navier_stokes][fsi]")
{
	json args = load_scene(
		std::string(POLYFEM_DATA_DIR) + "/standard/navier_stokes_fsi_moving_arc.json");
	args["time"] = {
		{"t0", 0},
		{"tend", 0.05},
		{"time_steps", 2},
		{"integrator", {{"type", "ImplicitEuler"}}}};
	args["/solver/linear/solver"_json_pointer] = "Eigen::SparseLU";

	State state;
	state.init(args, true);
	state.load_mesh();
	Eigen::MatrixXd solution;
	state.solve(solution);

	REQUIRE(state.variational_formulation != nullptr);
	CHECK(state.variational_formulation->name() == "NavierStokesFSI");
	CHECK(solution.allFinite());

	const test::NavierStokesFSIDebugData debug =
		test::VarFormTestAccess::navier_stokes_fsi_data(*state.variational_formulation);
	REQUIRE(debug.ale_form != nullptr);
	CHECK(debug.average_pressure_form == nullptr);
	const int mesh_offset = debug.velocity_ndof + debug.pressure_ndof;
	CHECK(solution.topRows(debug.velocity_ndof).norm() > 1e-8);
	CHECK(solution.middleRows(mesh_offset, debug.mesh_displacement_ndof).norm() > 1e-8);
	CHECK(debug.ale_form->is_step_valid(solution, solution));

	std::set<int> free_wall_nodes;
	for (const basis::ElementBases &element_bases : *debug.mesh_displacement_bases)
		for (const basis::Basis &basis : element_bases.bases)
			for (const auto &global : basis.global())
				if (std::abs(global.node(0) - 0.5) < 1e-8)
					free_wall_nodes.insert(global.index);
	REQUIRE_FALSE(free_wall_nodes.empty());
	double free_wall_displacement = 0;
	for (const int node : free_wall_nodes)
		free_wall_displacement = std::max(
			free_wall_displacement,
			solution.middleRows(mesh_offset + 2 * node, 2).norm());
	CHECK(free_wall_displacement > 1e-8);

	double reference_volume = 0;
	double current_volume = 0;
	for (int e = 0; e < int(debug.geometry_bases->size()); ++e)
	{
		assembler::ElementAssemblyValues displacement_vals;
		debug.mesh_displacement_cache->compute(
			e, debug.is_volume, debug.mesh_displacement_bases->at(e),
			debug.geometry_bases->at(e), displacement_vals);
		Eigen::VectorXd local_displacement = Eigen::VectorXd::Zero(
			int(displacement_vals.basis_values.size()) * 2);
		for (int a = 0; a < int(displacement_vals.basis_values.size()); ++a)
			for (int c = 0; c < 2; ++c)
				for (const auto &global : displacement_vals.basis_values[a].global)
					local_displacement(a * 2 + c) +=
						global.val * solution(mesh_offset + global.index * 2 + c);
		for (int q = 0; q < displacement_vals.quadrature.weights.size(); ++q)
		{
			Eigen::Matrix2d F = Eigen::Matrix2d::Identity();
			for (int a = 0; a < int(displacement_vals.basis_values.size()); ++a)
			{
				const Eigen::RowVector2d grad =
					displacement_vals.basis_values[a].grad.row(q) * displacement_vals.jac_it[q];
				for (int c = 0; c < 2; ++c)
					F.row(c) += local_displacement(a * 2 + c) * grad;
			}
			const double reference_weight =
				displacement_vals.det(q) * displacement_vals.quadrature.weights(q);
			reference_volume += reference_weight;
			current_volume += reference_weight * F.determinant();
		}
	}
	CHECK(std::abs(current_volume / reference_volume - 1) < 0.02);
}

TEST_CASE("periodic boundary conditions remain on legacy state path", "[varform][state]")
{
	json args = load_scene(std::string(POLYFEM_DATA_DIR) + "/standard/stokes_static.json");
	args["/boundary_conditions/periodic_boundary/enabled"_json_pointer] = true;

	CHECK_FALSE(varform::uses_varform_state(args));

	legacy::State state;
	state.init(args, false);

	REQUIRE(state.assembler != nullptr);
	CHECK(state.assembler->name() == "Stokes");
	REQUIRE(state.pressure_assembler != nullptr);
	CHECK(state.pressure_assembler->name() == "StokesPressure");
	CHECK(state.mixed_assembler != nullptr);
}

TEST_CASE("optimization keeps varforms on the legacy state path", "[varform][state]")
{
	json args = load_scene(std::string(POLYFEM_DATA_DIR) + "/standard/stokes_static.json");
	legacy::State state;
	state.optimization_enabled = true;
	state.init(args, true);

	REQUIRE(state.assembler != nullptr);
	CHECK(state.assembler->name() == "Stokes");
	REQUIRE(state.pressure_assembler != nullptr);
	CHECK(state.pressure_assembler->name() == "StokesPressure");
	CHECK(state.mixed_assembler != nullptr);
}
