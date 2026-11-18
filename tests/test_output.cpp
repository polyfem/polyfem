////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/io/OutStatsData.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/OutputWriter.hpp>
#include <polyfem/solver/forms/Form.hpp>
#include <polyfem/time_integrator/ImplicitEuler.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include "VarFormTestAccess.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <chrono>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::utils;

bool load_json(const std::string &json_file, json &out);

TEST_CASE("output interpolation preserves weighted mappings and sample order", "[output]")
{
	std::vector<basis::ElementBases> bases(2);
	for (auto &element : bases)
	{
		element.bases.resize(2);
		element.bases[0].set_basis([](const Eigen::MatrixXd &points, Eigen::MatrixXd &values) {
			values = Eigen::MatrixXd::Ones(points.rows(), 1) - points.col(0);
		});
		element.bases[1].set_basis([](const Eigen::MatrixXd &points, Eigen::MatrixXd &values) {
			values = points.col(0);
		});
	}
	const RowVectorNd node = RowVectorNd::Zero(1, 2);
	bases[0].bases[0].global().emplace_back(0, node, 1);
	bases[0].bases[1].global().emplace_back(1, node, 0.25);
	bases[0].bases[1].global().emplace_back(2, node, 0.75);
	bases[1].bases[0].global().emplace_back(2, node, 1);
	bases[1].bases[1].global().emplace_back(0, node, -0.5);
	bases[1].bases[1].global().emplace_back(2, node, 1.5);
	Eigen::MatrixXd points(7, 1);
	points << 0, 0.25, 0.5, 0.75, 1, 0.25, 0.75;
	Eigen::VectorXi elements(7);
	elements << 0, 0, 1, -1, 0, 1, 1;
	Eigen::MatrixXd expected(7, 3);
	expected << 1, 0, 0,
		0.75, 0.0625, 0.1875,
		-0.25, 0, 1.25,
		0, 0, 0,
		0, 0.25, 0.75,
		-0.125, 0, 1.125,
		-0.375, 0, 1.375;
	const StiffnessMatrix interpolation = io::Evaluator::build_interpolation_matrix(3, bases, points, elements);
	CHECK(Eigen::MatrixXd(interpolation).isApprox(expected, 1e-14));
}

TEST_CASE("elastic output reuses samples without changing fields", "[output]")
{
	const int dim = GENERATE(2, 3);
	const int order = GENERATE(1, 2);
	const std::string material = GENERATE("NeoHookean", "LinearElasticity", "HookeLinearElasticity", "SaintVenant",
										  "FixedCorotational", "MooneyRivlin3Param", "MooneyRivlin3ParamSymbolic", "InversionBarrier", "MaterialSum",
										  "MultiModels", "AMIPS", "NeoHookeanVarying");
	Eigen::MatrixXd vertices = Eigen::MatrixXd::Zero(dim + 2, dim);
	vertices.middleRows(1, dim).setIdentity();
	vertices.bottomRows(1).setOnes();
	Eigen::MatrixXi elements(2, dim + 1);
	for (int j = 0; j <= dim; ++j)
	{
		elements(0, j) = j;
		elements(1, j) = j + 1;
	}
	if (dim == 2)
		std::swap(elements(1, 1), elements(1, 2));
	json args;
	args["geometry"][0]["mesh"] = "";
	args["materials"] = {{"type", material}, {"E", 1e5}, {"nu", 0.3}};
	if (material == "NeoHookeanVarying")
		args["materials"] = {{"type", "NeoHookean"}, {"E", "100000*(1+0.1*x+0.2*y+0.05*z+0.03*t)"}, {"nu", 0.3}};
	if (material == "AMIPS")
		args["materials"] = {{"type", material}, {"use_rest_pose", true}};
	if (material == "MultiModels")
		args["materials"] = json::array({{{"id", 0}, {"type", "NeoHookean"}, {"E", 1e5}, {"nu", 0.3}},
										 {{"id", 1}, {"type", "LinearElasticity"}, {"E", 2e5}, {"nu", 0.25}}});
	if (material.find("MooneyRivlin") == 0)
		args["materials"] = {{"type", material}, {"c1", 2}, {"c2", 3}, {"c3", 1}, {"d1", 0.1}};
	if (material == "InversionBarrier")
		args["materials"]["JBarrierThreshold"] = 2;
	if (material == "MaterialSum")
		args["materials"] = {{"type", material}, {"models", json::array({{{"type", "NeoHookean"}, {"E", 1e5}, {"nu", 0.3}}, {{"type", "InversionBarrier"}, {"E", 1e5}, {"nu", 0.3}, {"JBarrierThreshold", 2}}})}};
	args["space"]["discr_order"] = order;
	args["contact"]["enabled"] = false;
	args["output"]["log"]["quiet"] = true;
	args["output"]["log"]["level"] = "error";
	State state;
	state.init(args, true);
	state.set_mesh(vertices, elements);
	auto &form = *state.variational_formulation;
	test::VarFormTestAccess::prepare(form);
	const auto debug = test::VarFormTestAccess::debug_data(form);
	Eigen::MatrixXd solution(debug.n_bases * dim, 1);
	for (int i = 0; i < solution.rows(); ++i)
		solution(i) = 0.001 * std::sin(i * 0.37);
	Eigen::MatrixXd nodes;
	if (dim == 2)
		autogen::p_nodes_2d(order, nodes);
	else
		autogen::p_nodes_3d(order, nodes);
	for (int layout = 0; layout < 3; ++layout)
	{
		CAPTURE(dim, order, layout, material);
		io::OutputSample sample;
		sample.local_points.resize(2 * nodes.rows() + 1, dim);
		sample.element_ids.resize(sample.local_points.rows());
		for (int e = 0; e < 2; ++e)
			for (int p = 0; p < nodes.rows(); ++p)
			{
				const int row = layout == 1 ? 2 * p + e : e * nodes.rows() + p;
				sample.local_points.row(row) = nodes.row(p);
				if (layout == 2)
					sample.local_points.row(row) = 0.7 * nodes.row(p) + Eigen::RowVectorXd::Constant(dim, 0.05);
				sample.element_ids(row) = e;
			}
		sample.local_points.bottomRows(1).setZero();
		sample.element_ids.tail(1).setConstant(-1);
		sample.points = sample.local_points;
		sample.time = 0.125;
		const auto fields = form.output_fields(sample, solution, {});
		Eigen::MatrixXd snapshot_solution = solution;
		const auto snapshot = test::VarFormTestAccess::snapshot_output_fields(form, snapshot_solution);
		REQUIRE(bool(snapshot));
		snapshot_solution.setConstant(42);
		const auto deferred_fields = snapshot(sample);
		REQUIRE(deferred_fields.size() == fields.size());
		for (size_t k = 0; k < fields.size(); ++k)
		{
			CHECK(deferred_fields[k].name == fields[k].name);
			CHECK(deferred_fields[k].association == fields[k].association);
			CHECK(deferred_fields[k].values.isApprox(fields[k].values));
		}
		const auto field_values = [&](const std::string &name) -> const Eigen::MatrixXd & {
			const auto it = std::find_if(fields.begin(), fields.end(), [&](const auto &field) { return field.name == name; });
			REQUIRE(it != fields.end());
			return it->values;
		};
		for (int row = 0; row < sample.local_points.rows() - 1; ++row)
		{
			const int e = sample.element_ids(row);
			const Eigen::MatrixXd point = sample.local_points.row(row);
			const assembler::OutputData data(sample.time, e, (*debug.bases)[e], (*debug.geometry_bases)[e], point, solution);
			std::vector<assembler::Assembler::NamedMatrix> scalar, tensor;
			const auto &elasticity = dynamic_cast<const assembler::ElasticityAssembler &>(*debug.assembler);
			Eigen::MatrixXd von_mises;
			elasticity.compute_von_mises_stresses(data, von_mises);
			scalar.emplace_back("von_mises", std::move(von_mises));
			const std::vector<std::pair<std::string, ElasticityTensorType>> tensor_types = {
				{"cauchy_stess", ElasticityTensorType::CAUCHY},
				{"pk1_stess", ElasticityTensorType::PK1},
				{"pk2_stess", ElasticityTensorType::PK2},
				{"F", ElasticityTensorType::F}};
			for (const auto &[name, type] : tensor_types)
			{
				Eigen::MatrixXd value;
				elasticity.compute_stress_tensor(data, type, value);
				tensor.emplace_back(name, std::move(value));
			}
			for (const auto &[name, value] : scalar)
				CHECK(field_values(name).row(row).isApprox(value, 1e-10));
			for (const auto &[name, value] : tensor)
				for (int d = 0; d < dim; ++d)
					CHECK(field_values(fmt::format("{}_{}", name, d + 1)).row(row).isApprox(value.middleCols(d * dim, dim), 1e-10));
		}
		const auto averaged = form.output_fields(sample, solution, {{"von_mises_avg", "cauchy_stess_avg", "pk1_stess_avg", "pk2_stess_avg", "F_avg"}});
		for (const auto &field : averaged)
			CHECK(field_values(field.name).isApprox(field.values, 1e-10));
		const auto gradients = form.output_fields(sample, solution, {{"solution", "solution_gradient"}});
		for (int row = 0; row < sample.local_points.rows() - 1; ++row)
		{
			Eigen::MatrixXd value, gradient;
			io::Evaluator::interpolate_at_local_vals(*debug.mesh, dim, *debug.bases, *debug.geometry_bases,
													 sample.element_ids(row), sample.local_points.row(row), solution, value, gradient);
			REQUIRE(gradients.size() == 2);
			CHECK(gradients[0].values.row(row).isApprox(value, 1e-12));
			CHECK(gradients[1].values.row(row).isApprox(gradient, 1e-12));
		}
		for (const auto &field : fields)
			CHECK(field.values.bottomRows(1).isZero());
	}
}

TEST_CASE("output statistics are initialized", "[output]")
{
	io::OutRuntimeData timings;
	CHECK(timings.total_time() == 0);
	CHECK(timings.loading_mesh_time == 0);
	CHECK(timings.assigning_rhs_time == 0);
	timings.building_basis_time = 1;
	timings.loading_mesh_time = 2;
	timings.assigning_rhs_time = 3;
	timings.solving_time = 4;
	CHECK(timings.total_time() == 10);

	io::OutStatsData stats;
	CHECK(stats.spectrum.isZero());
	CHECK(stats.mesh_size == 0);
	CHECK(stats.min_edge_length == 0);
	CHECK(stats.nn_zero == 0);
	CHECK(stats.simplex_count == 0);
}

TEST_CASE("absolute AMIPS output preserves undefined stresses", "[output]")
{
	Eigen::MatrixXd vertices(3, 2);
	vertices << 0, 0, 1, 0, 0, 1;
	Eigen::MatrixXi elements(1, 3);
	elements << 0, 1, 2;
	json args;
	args["geometry"][0]["mesh"] = "";
	args["materials"] = {{"type", "AMIPS"}, {"use_rest_pose", false}};
	args["contact"]["enabled"] = false;
	args["output"]["log"]["quiet"] = true;
	args["output"]["log"]["level"] = "error";
	State state;
	state.init(args, true);
	state.set_mesh(vertices, elements);
	auto &form = *state.variational_formulation;
	test::VarFormTestAccess::prepare(form);
	const auto debug = test::VarFormTestAccess::debug_data(form);
	const Eigen::MatrixXd solution = Eigen::MatrixXd::Zero(2 * debug.n_bases, 1);
	io::OutputSample sample;
	autogen::p_nodes_2d(1, sample.local_points);
	sample.points = sample.local_points;
	sample.element_ids = Eigen::VectorXi::Zero(sample.local_points.rows());
	for (const auto &field : form.output_fields(sample, solution, {}))
	{
		CAPTURE(field.name);
		if (field.name.find("stess") != std::string::npos || field.name.find("von_mises") != std::string::npos)
			CHECK(field.values.array().isNaN().all());
		else
			CHECK(field.values.allFinite());
	}
}

TEST_CASE("elastic output captures changing integrator and force data", "[output_writer]")
{
	class MutableForce : public solver::Form
	{
	public:
		double derivative = 0.3;
		std::string name() const override { return "test"; }
		double value_unweighted(const Eigen::VectorXd &) const override { return 0; }
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradient) const override
		{
			gradient.setConstant(x.size(), derivative);
		}
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override
		{
			hessian.resize(x.size(), x.size());
			hessian.setZero();
		}
	};
	Eigen::MatrixXd vertices(3, 2);
	vertices << 0, 0, 1, 0, 0, 1;
	Eigen::MatrixXi elements(1, 3);
	elements << 0, 1, 2;
	json args;
	args["geometry"][0]["mesh"] = "";
	args["materials"] = {{"type", "NeoHookean"}, {"E", 1e5}, {"nu", 0.3}, {"rho", 1}};
	args["time"] = {{"dt", 0.01}, {"tend", 0.02}};
	args["contact"]["enabled"] = false;
	args["output"]["log"]["quiet"] = true;
	args["output"]["log"]["level"] = "error";
	args["output"]["paraview"]["options"] = {{"velocity", true}, {"acceleration", true}, {"forces", true}};
	args["output"]["paraview"]["fields"] = {"velocity", "acceleration", "test_forces"};
	State state;
	state.init(args, true);
	state.set_mesh(vertices, elements);
	auto &form = dynamic_cast<varform::ElasticVarForm &>(*state.variational_formulation);
	test::VarFormTestAccess::prepare(form);
	const int ndof = 2 * test::VarFormTestAccess::debug_data(form).n_bases;
	const Eigen::VectorXd solution = Eigen::VectorXd::Zero(ndof);
	time_integrator::ImplicitEuler integrator;
	integrator.init(solution, Eigen::VectorXd::Constant(ndof, 0.4), Eigen::VectorXd::Constant(ndof, 0.2), 0.01);
	auto force = std::make_shared<MutableForce>();
	const double expected_force = -force->derivative / integrator.acceleration_scaling();
	const auto snapshot = test::VarFormTestAccess::snapshot_elastic_fields(form, solution, &integrator, {{"test", force}});
	force->derivative = 42;
	integrator.update_quantities(Eigen::VectorXd::Ones(ndof));
	io::OutputSample sample;
	autogen::p_nodes_2d(1, sample.local_points);
	sample.points = sample.local_points;
	sample.element_ids = Eigen::VectorXi::Zero(sample.local_points.rows());
	std::vector<io::OutputField> fields;
	io::OutputWriter writer;
	writer.enqueue([&] { return [&] { fields = snapshot(sample); }; });
	writer.finish();
	REQUIRE(fields.size() == 3);
	const std::map<std::string, double> expected_values{
		{"velocity", 0.4}, {"acceleration", 0.2}, {"test_forces", expected_force}};
	for (const auto &field : fields)
	{
		CAPTURE(field.name);
		CHECK((field.values.array() - expected_values.at(field.name)).abs().maxCoeff() < 1e-10);
	}
}

TEST_CASE("output writer preserves order and drains", "[output_writer]")
{
	const bool asynchronous = GENERATE(false, true);
	std::vector<int> written;
	{
		io::OutputWriter writer(asynchronous);
		for (int i = 0; i < 20; ++i)
			writer.enqueue([&, i] { return [&, i] { written.push_back(i); }; });
		writer.flush();
		REQUIRE(written.size() == 20);
		writer.enqueue([&] { return [&] { written.push_back(20); }; });
	}
	REQUIRE(written.size() == 21);
	for (int i = 0; i < written.size(); ++i)
		CHECK(written[i] == i);
}

TEST_CASE("output writer blocks preparation with one pending frame", "[output_writer]")
{
	io::OutputWriter writer;
	if (!writer.asynchronous())
		return;
	using namespace std::chrono_literals;
	std::promise<void> active, release, producer_started, prepared;
	auto active_future = active.get_future();
	auto release_future = release.get_future().share();
	auto started_future = producer_started.get_future();
	auto prepared_future = prepared.get_future();
	writer.enqueue([&] { return [&] { active.set_value(); release_future.wait(); }; });
	const bool running = active_future.wait_for(5s) == std::future_status::ready;
	if (!running)
	{
		release.set_value();
		writer.finish();
		FAIL("Output worker did not start");
	}
	writer.enqueue([] { return [] {}; });
	auto producer = std::async(std::launch::async, [&] {
		producer_started.set_value();
		writer.enqueue([&] { prepared.set_value(); return [] {}; });
	});
	started_future.wait();
	const bool blocked = prepared_future.wait_for(50ms) == std::future_status::timeout;
	release.set_value();
	producer.get();
	writer.finish();
	CHECK(blocked);
	CHECK(prepared_future.wait_for(0ms) == std::future_status::ready);
}

TEST_CASE("serial output waits for an active write before finishing", "[output_writer]")
{
	io::OutputWriter writer(false);
	using namespace std::chrono_literals;
	std::promise<void> active, release, finishing, finished;
	auto active_future = active.get_future();
	auto release_future = release.get_future().share();
	auto finishing_future = finishing.get_future();
	auto finished_future = finished.get_future();
	auto producer = std::async(std::launch::async, [&] {
		writer.enqueue([&] { return [&] { active.set_value(); release_future.wait(); }; });
	});
	active_future.wait();
	auto closer = std::async(std::launch::async, [&] {
		finishing.set_value();
		writer.finish();
		finished.set_value();
	});
	finishing_future.wait();
	const bool blocked = finished_future.wait_for(50ms) == std::future_status::timeout;
	release.set_value();
	producer.get();
	closer.get();
	CHECK(blocked);
}

TEST_CASE("output writer propagates failures and closes", "[output_writer]")
{
	const bool asynchronous = GENERATE(false, true);
	io::OutputWriter writer(asynchronous);
	const auto prepare = []() -> io::OutputWriter::Job {
		return [] { throw std::runtime_error("output failure"); };
	};
	if (writer.asynchronous())
	{
		writer.enqueue(prepare);
		REQUIRE_THROWS_WITH(writer.flush(), "output failure");
	}
	else
		REQUIRE_THROWS_WITH(writer.enqueue(prepare), "output failure");
	REQUIRE_THROWS_WITH(writer.finish(), "output failure");
	REQUIRE_THROWS_WITH(writer.enqueue(prepare), "output failure");
}

TEST_CASE("output writer can recover from frame preparation failure", "[output_writer]")
{
	const bool asynchronous = GENERATE(false, true);
	io::OutputWriter writer(asynchronous);
	REQUIRE_THROWS_WITH(writer.enqueue([]() -> io::OutputWriter::Job {
		throw std::runtime_error("preparation failure");
	}),
						"preparation failure");
	bool written = false;
	writer.enqueue([&] { return [&] { written = true; }; });
	writer.finish();
	writer.finish();
	CHECK(written);
	REQUIRE_THROWS_AS(writer.enqueue([] { return [] {}; }), std::logic_error);
}

#ifdef NDEBUG
TEST_CASE("full sim", "[full_sim]")
#else
TEST_CASE("full sim", "[.][full_sim]")
#endif
{
	const std::string scene = GENERATE("2D/unit-tests/5-squares", "3D/unit-tests/5-cubes-fast", "3D/unit-tests/edge-edge-parallel");
	const std::string scene_file = fmt::format("{}/contact/examples/{}.json", POLYFEM_DATA_DIR, scene);

	json args;
	if (!load_json(scene_file, args))
	{
		spdlog::error("unable to open {} file", scene_file);
		FAIL();
	}

	args["root_path"] = scene_file;
	const std::filesystem::path outdir = std::filesystem::current_path() / "DELETE_ME_full_sim_test_output";
	args["/output/directory"_json_pointer] = outdir.string();
	args["/output/paraview"_json_pointer] = R"({
		"file_name": "sim.pvd",
		"volume": true,
		"surface": true,
		"wireframe": true,
		"points": true,
		"vismesh_rel_area": 1e-05,
		"options": {
			"material": true,
			"body_ids": true,
			"contact_forces": true,
			"friction_forces": true,
			"velocity": true,
			"acceleration": true
		}
	})"_json;
	args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
	args["/output/log/level"_json_pointer] = "warning";

	State state;

	state.init(args, true);
	state.load_mesh();

	Eigen::MatrixXd sol;

	state.solve(sol);
	state.variational_formulation->compute_errors(sol);

	CHECK(std::filesystem::exists(outdir));
	CHECK(std::filesystem::exists(outdir / "sim.pvd"));

	int n_volume_files = 0;
	for (const auto &entry : std::filesystem::directory_iterator(outdir))
	{
		if (!std::regex_match(entry.path().filename().string(), std::regex(R"(step_\d+\.vtu)")))
			continue;
		std::ifstream file(entry.path());
		const std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		CAPTURE(entry.path().filename().string());
		CHECK(contents.find("Name=\"solution\"") != std::string::npos);
		++n_volume_files;
	}
	CHECK(n_volume_files > 0);

	std::filesystem::remove_all(outdir);
}
