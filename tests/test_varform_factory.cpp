#include <polyfem/State.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>
#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/NavierStokesFSIForm.hpp>

#include "VarFormTestAccess.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <fstream>
#include <filesystem>
#include <iterator>
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
	CHECK_FALSE(varform::VarFormFactory::supports("UnknownFormulation", args));
	CHECK(varform::VarFormFactory::create("UnknownFormulation", args) == nullptr);

	json contact_fsi_args = args;
	contact_fsi_args["contact"]["enabled"] = true;
	contact_fsi_args["materials"] = {
		{"type", "NavierStokesFSI"},
		{"fluid_geometry_id", 1},
		{"solid_geometry_id", 2},
		{"displacement_space_id", 3},
		{"solid_material", {{"type", "NeoHookean"}}}};
	CHECK(varform::VarFormFactory::supports("NavierStokesFSI", contact_fsi_args));
	CHECK(varform::VarFormFactory::create("NavierStokesFSI", contact_fsi_args) != nullptr);
	CHECK(varform::uses_varform_state(contact_fsi_args));

	contact_fsi_args["materials"].erase("solid_material");
	CHECK_FALSE(varform::VarFormFactory::supports("NavierStokesFSI", contact_fsi_args));
	CHECK(varform::VarFormFactory::create("NavierStokesFSI", contact_fsi_args) == nullptr);

	json static_args = args;
	static_args["time"] = nullptr;
	CHECK_FALSE(varform::VarFormFactory::supports("NavierStokesFSI", static_args));
	CHECK(varform::VarFormFactory::create("NavierStokesFSI", static_args) == nullptr);

	json boundary_pair_args = args;
	boundary_pair_args["/boundary_conditions/periodic"_json_pointer] = {{{"boundary_ids", {1, 2}}}};
	CHECK(varform::VarFormFactory::supports("NeoHookean", boundary_pair_args));
	CHECK(varform::VarFormFactory::create("NeoHookean", boundary_pair_args) != nullptr);
	const auto scalar_with_periodic = varform::VarFormFactory::create("Laplacian", boundary_pair_args);
	REQUIRE(scalar_with_periodic != nullptr);
	CHECK(scalar_with_periodic->name() == "Scalar");
	const auto linear_with_periodic = varform::VarFormFactory::create("LinearElasticity", boundary_pair_args);
	REQUIRE(linear_with_periodic != nullptr);
	CHECK(linear_with_periodic->name() == "NonlinearElasticTransient");

	json periodic_homogenization_args = boundary_pair_args;
	periodic_homogenization_args["time"] = nullptr;
	periodic_homogenization_args["/constraints/macro_displacement_gradient"_json_pointer] = {
		{"value", {{0, 0}, {0, 0}}},
		{"fixed_components", {0}}};
	CHECK(varform::VarFormFactory::supports("NeoHookean", periodic_homogenization_args, true));
	const auto differentiable_periodic =
		varform::VarFormFactory::create("NeoHookean", periodic_homogenization_args, true);
	REQUIRE(differentiable_periodic != nullptr);
	CHECK(std::dynamic_pointer_cast<varform::DifferentiableVarForm>(differentiable_periodic) != nullptr);
	CHECK(differentiable_periodic->name() == "NonlinearElasticStatic");

	periodic_homogenization_args["/contact/periodic"_json_pointer] = true;
	periodic_homogenization_args["/contact/enabled"_json_pointer] = true;
	CHECK(varform::VarFormFactory::supports("NeoHookean", periodic_homogenization_args, true));
	CHECK(varform::VarFormFactory::create("NeoHookean", periodic_homogenization_args, true) != nullptr);

	json zero_mean_args = args;
	zero_mean_args["/constraints/zero_mean"_json_pointer] = true;
	const auto scalar_with_zero_mean = varform::VarFormFactory::create("Laplacian", zero_mean_args);
	REQUIRE(scalar_with_zero_mean != nullptr);
	CHECK(scalar_with_zero_mean->name() == "Scalar");
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

TEST_CASE("coupled two-mesh Navier-Stokes FSI", "[varform][state][navier_stokes][fsi]")
{
	const std::filesystem::path output_directory =
		std::filesystem::temp_directory_path() / "polyfem-two-mesh-fsi";
	std::filesystem::remove_all(output_directory);
	json args = load_scene(
		std::string(POLYFEM_DATA_DIR) + "/standard/navier_stokes_fsi_simple_square.json");
	args["time"] = {
		{"t0", 0},
		{"tend", 0.01},
		{"time_steps", 1},
		{"integrator", args["time"]["integrator"]}};
	args["/solver/linear/solver"_json_pointer] = "Eigen::SparseLU";
	args["/output/directory"_json_pointer] = output_directory.string();
	args["/output/advanced/save_time_sequence"_json_pointer] = true;
	args["/output/paraview/file_name"_json_pointer] = "fsi.pvd";
	args["/output/paraview/surface"_json_pointer] = true;
	args["/output/data/state"_json_pointer] =
		(output_directory / "state-{:d}.h5").string();

	State state;
	state.init(args, true);
	state.load_mesh();
	Eigen::MatrixXd solution;
	state.solve(solution);

	REQUIRE(state.variational_formulation != nullptr);
	CHECK(solution.allFinite());
	const test::NavierStokesFSIDebugData debug =
		test::VarFormTestAccess::navier_stokes_fsi_data(*state.variational_formulation);
	REQUIRE(debug.problem != nullptr);
	REQUIRE(debug.solid_varform != nullptr);
	REQUIRE(debug.fluid_mesh != nullptr);
	REQUIRE(debug.solid_mesh != nullptr);
	REQUIRE(debug.interface_form != nullptr);
	REQUIRE(debug.average_pressure_form != nullptr);
	CHECK(debug.fluid_mesh->has_geometry_ids());
	CHECK(debug.solid_mesh->has_geometry_ids());
	CHECK(debug.interface_size > 0);
	CHECK(debug.solid_displacement_offset
		  == debug.velocity_ndof + debug.pressure_ndof + debug.mesh_displacement_ndof);
	CHECK(debug.problem->full_size()
		  == debug.solid_displacement_offset + debug.solid_displacement_ndof
				 + debug.fluid_multiplier_ndof + debug.mesh_multiplier_ndof
				 + (debug.average_pressure_form ? 1 : 0));
	CHECK(debug.fluid_multiplier_offset
		  == debug.solid_displacement_offset + debug.solid_displacement_ndof);
	CHECK(debug.mesh_multiplier_offset
		  == debug.fluid_multiplier_offset + debug.fluid_multiplier_ndof);
	CHECK(debug.average_pressure_offset
		  == debug.mesh_multiplier_offset + debug.mesh_multiplier_ndof);
	Eigen::VectorXd pressure_gauge_residual;
	debug.average_pressure_form->first_derivative(solution.col(0), pressure_gauge_residual);
	CHECK(std::abs(pressure_gauge_residual(debug.average_pressure_offset)
				   / debug.average_pressure_form->weight())
		  < 1e-6);

	const int mesh_offset = debug.velocity_ndof + debug.pressure_ndof;
	CHECK(solution.topRows(debug.velocity_ndof).norm() > 1e-8);
	CHECK(solution.middleRows(mesh_offset, debug.mesh_displacement_ndof).norm() > 1e-10);
	CHECK(solution.middleRows(
					  debug.solid_displacement_offset, debug.solid_displacement_ndof)
			  .norm()
		  > 1e-8);

	const Eigen::VectorXd solid = solution.middleRows(
		debug.solid_displacement_offset, debug.solid_displacement_ndof);
	CHECK(debug.interface_form->physical_constraint(
								  solution.topRows(debug.velocity_ndof),
								  debug.solid_varform->embedding_time_integrator()->v_prev())
			  .norm()
		  < 1e-5);
	CHECK(debug.interface_form->mesh_constraint(
								  solution.middleRows(mesh_offset, debug.mesh_displacement_ndof), solid)
			  .norm()
		  < 1e-5);

	StiffnessMatrix interface_jacobian;
	debug.interface_form->second_derivative(solution.col(0), interface_jacobian);
	CHECK(Eigen::MatrixXd(interface_jacobian.block(
							  debug.solid_displacement_offset, debug.mesh_multiplier_offset,
							  debug.solid_displacement_ndof, debug.mesh_multiplier_ndof))
			  .isZero(0));
	CHECK(Eigen::MatrixXd(interface_jacobian.block(
							  mesh_offset, debug.mesh_multiplier_offset,
							  debug.mesh_displacement_ndof, debug.mesh_multiplier_ndof))
			  .norm()
		  > 0);
	Eigen::VectorXd interface_direction = Eigen::VectorXd::LinSpaced(
		solution.rows(), -0.7, 0.9);
	interface_direction.normalize();
	const double interface_eps = 1e-7;
	Eigen::VectorXd interface_plus, interface_minus;
	debug.interface_form->first_derivative(
		solution.col(0) + interface_eps * interface_direction, interface_plus);
	debug.interface_form->first_derivative(
		solution.col(0) - interface_eps * interface_direction, interface_minus);
	CHECK(Eigen::VectorXd(interface_jacobian * interface_direction)
			  .isApprox((interface_plus - interface_minus) / (2 * interface_eps), 1e-8));

	const int multiplier_offset = debug.average_pressure_offset;
	solver::NavierStokesFSIAveragePressureForm embedded_gauge(
		multiplier_offset + 1,
		debug.velocity_ndof / debug.fluid_mesh->dimension(), debug.pressure_ndof,
		debug.mesh_displacement_ndof / debug.fluid_mesh->dimension(),
		multiplier_offset, debug.fluid_mesh->dimension(),
		*debug.pressure_bases, *debug.mesh_displacement_bases, *debug.geometry_bases,
		*debug.pressure_cache, *debug.mesh_displacement_cache, debug.is_volume);
	Eigen::VectorXd gauge_x = Eigen::VectorXd::Zero(multiplier_offset + 1);
	gauge_x.segment(debug.velocity_ndof, debug.pressure_ndof) =
		Eigen::VectorXd::LinSpaced(debug.pressure_ndof, -0.2, 0.3);
	gauge_x(multiplier_offset) = 0.4;
	StiffnessMatrix gauge_jacobian;
	embedded_gauge.second_derivative(gauge_x, gauge_jacobian);
	bool gauge_touches_solid = false;
	for (int col = 0; col < gauge_jacobian.outerSize(); ++col)
		for (StiffnessMatrix::InnerIterator entry(gauge_jacobian, col); entry; ++entry)
		{
			const bool row_is_solid = entry.row() >= debug.solid_displacement_offset
									  && entry.row() < debug.solid_displacement_offset + debug.solid_displacement_ndof;
			const bool col_is_solid = entry.col() >= debug.solid_displacement_offset
									  && entry.col() < debug.solid_displacement_offset + debug.solid_displacement_ndof;
			gauge_touches_solid |= row_is_solid || col_is_solid;
		}
	CHECK_FALSE(gauge_touches_solid);

	Eigen::Vector2d mean_solid_displacement = Eigen::Vector2d::Zero();
	for (int node = 0; node < solid.size() / 2; ++node)
		mean_solid_displacement += solid.segment<2>(2 * node);
	mean_solid_displacement /= solid.size() / 2;
	// The current validation scene has no solid body force: the inflow pushes
	// the solid downstream through the coupled interface.
	CHECK(mean_solid_displacement.x() > 1e-8);

	Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(
		debug.solid_displacement_ndof, -0.5, 0.7);
	direction.normalize();
	Eigen::VectorXd residual = Eigen::VectorXd::Zero(debug.solid_displacement_ndof);
	StiffnessMatrix solid_jacobian(debug.solid_displacement_ndof, debug.solid_displacement_ndof);
	for (const auto &form : debug.solid_varform->embedding_forms())
	{
		Eigen::VectorXd form_residual;
		StiffnessMatrix form_jacobian;
		form->first_derivative(solid, form_residual);
		form->second_derivative(solid, form_jacobian);
		residual += form_residual;
		solid_jacobian += form_jacobian;
	}
	const double eps = 1e-7;
	Eigen::VectorXd plus_residual = Eigen::VectorXd::Zero(debug.solid_displacement_ndof);
	Eigen::VectorXd minus_residual = Eigen::VectorXd::Zero(debug.solid_displacement_ndof);
	for (const auto &form : debug.solid_varform->embedding_forms())
	{
		Eigen::VectorXd form_residual;
		form->first_derivative(solid + eps * direction, form_residual);
		plus_residual += form_residual;
		form->first_derivative(solid - eps * direction, form_residual);
		minus_residual += form_residual;
	}
	const Eigen::VectorXd finite_difference = (plus_residual - minus_residual) / (2 * eps);
	CHECK(Eigen::VectorXd(solid_jacobian * direction).isApprox(finite_difference, 2e-5));

	CHECK(std::filesystem::is_regular_file(output_directory / "fsi.pvd"));
	CHECK_FALSE(std::filesystem::exists(output_directory / "solid_fsi.pvd"));
	CHECK(std::filesystem::is_regular_file(output_directory / "step_0.vtm"));
	CHECK_FALSE(std::filesystem::exists(output_directory / "solid_step_0.vtm"));
	const auto read_text = [](const std::filesystem::path &path) {
		std::ifstream input(path);
		return std::string(
			std::istreambuf_iterator<char>(input),
			std::istreambuf_iterator<char>());
	};
	const std::string fluid_pvd = read_text(output_directory / "fsi.pvd");
	CHECK(fluid_pvd.find("step_0.vtm") != std::string::npos);
	CHECK(fluid_pvd.find("solid_step_") == std::string::npos);
	const std::string combined_vtm = read_text(output_directory / "step_0.vtm");
	CHECK(combined_vtm.find("step_0.vtu") != std::string::npos);
	CHECK(combined_vtm.find("step_0_surf.vtu") != std::string::npos);
	CHECK(combined_vtm.find("solid_step_0.vtu") != std::string::npos);
	CHECK(combined_vtm.find("solid_step_0_surf.vtu") != std::string::npos);
	CHECK(combined_vtm.find("Fluid Volume") != std::string::npos);
	CHECK(combined_vtm.find("Fluid Surface") != std::string::npos);
	CHECK(combined_vtm.find("Solid Volume") != std::string::npos);
	CHECK(combined_vtm.find("Solid Surface") != std::string::npos);
	const std::string state_path = (output_directory / "state-1.h5").string();
	for (const std::string &name : {"solid_u", "solid_v", "solid_a"})
	{
		Eigen::MatrixXd history;
		CHECK(io::read_matrix(state_path, name, history));
		CHECK(history.rows() == debug.solid_displacement_ndof);
		CHECK(history.allFinite());
	}
	std::filesystem::remove_all(output_directory);
}

TEST_CASE("macro displacement gradient remains on legacy state path", "[varform][state]")
{
	json args = load_scene(std::string(POLYFEM_DATA_DIR) + "/standard/neohookean.json");
	args["/constraints/macro_displacement_gradient"_json_pointer] = {
		{"value", {{0, 0}, {0, 0}}},
		{"fixed_components", {0}}};

	CHECK_FALSE(varform::uses_varform_state(args));

	legacy::State state;
	state.init(args, false);

	REQUIRE(state.assembler != nullptr);
	CHECK(state.assembler->name() == "NeoHookean");
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
