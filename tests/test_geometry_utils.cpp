#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

#include <filesystem>
#include <fstream>

namespace
{
	using polyfem::json;

	json identity_transform()
	{
		return json{
			{"dimensions", nullptr},
			{"scale", json::array()},
			{"rotation", nullptr},
			{"rotation_mode", "xyz"},
			{"translation", json::array()}};
	}

	json obstacle_mesh_json(const std::filesystem::path &mesh_path, const std::string &extract)
	{
		return json{
			{"mesh", mesh_path.string()},
			{"unit", ""},
			{"extract", extract},
			{"n_refs", 0},
			{"advanced", {{"refinement_location", 0.5}}},
			{"transformation", identity_transform()}};
	}

	json fem_mesh_json(const std::filesystem::path &mesh_path)
	{
		return json{
			{"mesh", mesh_path.string()},
			{"unit", ""},
			{"extract", "volume"},
			{"n_refs", 0},
			{"advanced", {{"normalize_mesh", false}, {"refinement_location", 0.5}, {"min_component", -1}, {"force_linear_geometry", false}}},
			{"transformation", identity_transform()},
			{"curve_selection", nullptr},
			{"volume_selection", {{"id_offset", 0}}}};
	}

	std::filesystem::path write_geometry_reader_obj(const std::string &name, const std::string &contents)
	{
		const std::filesystem::path path = std::filesystem::temp_directory_path() / ("polyfem_geometry_reader_" + name);
		std::ofstream out(path);
		REQUIRE(out.good());
		out << contents;
		return path;
	}

	void require_matrix_approx(const Eigen::MatrixXd &actual, const Eigen::MatrixXd &expected, const double margin = 1e-12)
	{
		REQUIRE(actual.rows() == expected.rows());
		REQUIRE(actual.cols() == expected.cols());
		for (int r = 0; r < actual.rows(); ++r)
			for (int c = 0; c < actual.cols(); ++c)
				CHECK(actual(r, c) == Catch::Approx(expected(r, c)).margin(margin));
	}
} // namespace

TEST_CASE("geometry reader handles stored Gmsh surface selections", "[geometry][gmsh]")
{
	using namespace polyfem;
	using namespace polyfem::mesh;

	const Units units;
	const std::filesystem::path mesh_path = std::filesystem::path(POLYFEM_DATA_DIR) / "gmsh_physical_sides_2d_v22.msh";

	SECTION("explicit surface selections override imported tags")
	{
		json args = fem_mesh_json(mesh_path);
		args["surface_selection"] = 77;
		const auto mesh = read_fem_mesh(units, args, "");
		REQUIRE(mesh != nullptr);
		for (int e = 0; e < mesh->n_edges(); ++e)
			CHECK(mesh->get_boundary_id(e) == (mesh->is_boundary_edge(e) ? 77 : -1));
	}

	SECTION("refinement requires an explicit replacement selection")
	{
		json args = fem_mesh_json(mesh_path);
		args["n_refs"] = 1;
		REQUIRE_THROWS_WITH(
			read_fem_mesh(units, args, ""),
			Catch::Matchers::ContainsSubstring("stored surface selections"));
	}
}

TEST_CASE("Triangle area", "[geometry]")
{
	using namespace polyfem::utils;

	Eigen::Matrix3d V;
	V << 0, 0, 0,
		1, 0, 0,
		0, 1, 0;

	CHECK(triangle_area(V.leftCols<2>()) == Catch::Approx(0.5));
	CHECK(triangle_area(V) == Catch::Approx(0.5));

	Eigen::Matrix3d V_flipped = V;
	V_flipped.row(1) = V.row(2);
	V_flipped.row(2) = V.row(1);

	CHECK(triangle_area(V_flipped.leftCols<2>()) == Catch::Approx(-0.5));
	CHECK(triangle_area(V_flipped) == Catch::Approx(0.5));
}

TEST_CASE("Tetrahedron volume", "[geometry]")
{
	using namespace polyfem::utils;

	Eigen::Matrix<double, 4, 3> V;
	V << 0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;

	CHECK(tetrahedron_volume(V) == Catch::Approx(1 / 6.));

	Eigen::Matrix<double, 4, 3> V_flipped = V;
	V_flipped.row(2) = V.row(3);
	V_flipped.row(3) = V.row(2);

	CHECK(tetrahedron_volume(V_flipped) == Catch::Approx(-1 / 6.));
}

TEST_CASE("geometry reader affine transformations", "[geometry][geometry_reader]")
{
	using namespace polyfem;
	using namespace polyfem::mesh;

	MatrixNd A;
	VectorNd b;

	VectorNd dims2(2);
	dims2 << 2.0, 4.0;
	json transform2d = identity_transform();
	transform2d["scale"] = json::array({2.0, 3.0});
	transform2d["rotation"] = 90.0;
	transform2d["translation"] = json::array({5.0});
	construct_affine_transformation(1.0, transform2d, dims2, A, b);

	Eigen::Matrix2d expected_A2;
	expected_A2 << 0.0, -3.0,
		2.0, 0.0;
	Eigen::Vector2d expected_b2;
	expected_b2 << 5.0, 0.0;
	require_matrix_approx(A, expected_A2, 1e-12);
	require_matrix_approx(b, expected_b2, 1e-12);

	VectorNd dims3(3);
	dims3 << 2.0, 0.0, 4.0;
	json dimensions_transform = identity_transform();
	dimensions_transform["dimensions"] = json::array({4.0, 6.0});
	dimensions_transform["rotation"] = json::array();
	dimensions_transform["translation"] = json::array({1.0, 2.0, 3.0});
	construct_affine_transformation(0.5, dimensions_transform, dims3, A, b);

	Eigen::Matrix3d expected_dimensions_A = Eigen::Matrix3d::Zero();
	expected_dimensions_A.diagonal() << 1.0, 3.0, 0.0;
	Eigen::Vector3d expected_b3;
	expected_b3 << 1.0, 2.0, 3.0;
	require_matrix_approx(A, expected_dimensions_A, 1e-12);
	require_matrix_approx(b, expected_b3, 1e-12);

	json axis_angle_transform = identity_transform();
	axis_angle_transform["scale"] = 2.0;
	axis_angle_transform["rotation"] = json::array({90.0, 0.0, 0.0, 1.0});
	axis_angle_transform["rotation_mode"] = "axis_angle";
	axis_angle_transform["translation"] = json::array({0.0, 0.0, 0.0});
	construct_affine_transformation(1.0, axis_angle_transform, dims3, A, b);

	Eigen::Matrix3d expected_axis_A;
	expected_axis_A << 0.0, -2.0, 0.0,
		2.0, 0.0, 0.0,
		0.0, 0.0, 2.0;
	require_matrix_approx(A, expected_axis_A, 1e-12);
	CHECK(b.isZero(1e-12));
}

TEST_CASE("geometry reader obstacle mesh extraction modes", "[geometry][geometry_reader]")
{
	using namespace polyfem;
	using namespace polyfem::mesh;

	const Units units;
	const std::filesystem::path line_path = write_geometry_reader_obj(
		"line.obj",
		"v 0 0 0\n"
		"v 1 0 0\n"
		"l 1 2\n");
	const std::filesystem::path tri_path = write_geometry_reader_obj(
		"tri.obj",
		"v 0 0 0\n"
		"v 1 0 0\n"
		"v 0 1 0\n"
		"f 1 2 3\n");

	Eigen::MatrixXd vertices;
	Eigen::VectorXi codim_vertices;
	Eigen::MatrixXi codim_edges;
	Eigen::MatrixXi faces;

	json points_mesh = obstacle_mesh_json(line_path, "points");
	read_obstacle_mesh(units, points_mesh, "", 2, vertices, codim_vertices, codim_edges, faces);
	REQUIRE(vertices.rows() == 2);
	REQUIRE(vertices.cols() == 2);
	CHECK(codim_vertices.size() == 2);
	CHECK(codim_edges.size() == 0);
	CHECK(faces.size() == 0);

	json refined_edges_mesh = obstacle_mesh_json(line_path, "edges");
	refined_edges_mesh["n_refs"] = 1;
	refined_edges_mesh["advanced"]["refinement_location"] = 0.25;
	refined_edges_mesh["transformation"]["scale"] = 2.0;
	refined_edges_mesh["transformation"]["translation"] = json::array({1.0, 0.0});
	read_obstacle_mesh(units, refined_edges_mesh, "", 2, vertices, codim_vertices, codim_edges, faces);
	REQUIRE(vertices.rows() == 3);
	REQUIRE(codim_edges.rows() == 2);
	CHECK(faces.size() == 0);
	CHECK(vertices(0, 0) == Catch::Approx(1.0));
	CHECK(vertices(1, 0) == Catch::Approx(3.0));
	CHECK(vertices(2, 0) == Catch::Approx(1.5));
	CHECK(codim_edges.row(0).isApprox(Eigen::RowVector2i(0, 2)));
	CHECK(codim_edges.row(1).isApprox(Eigen::RowVector2i(2, 1)));

	json edge_from_surface_mesh = obstacle_mesh_json(tri_path, "edges");
	read_obstacle_mesh(units, edge_from_surface_mesh, "", 2, vertices, codim_vertices, codim_edges, faces);
	CHECK(codim_edges.rows() == 3);
	CHECK(faces.size() == 0);

	json surface_in_2d_mesh = obstacle_mesh_json(tri_path, "surface");
	read_obstacle_mesh(units, surface_in_2d_mesh, "", 2, vertices, codim_vertices, codim_edges, faces);
	CHECK(codim_edges.rows() == 3);
	CHECK(faces.size() == 0);

	json volume_obstacle = obstacle_mesh_json(line_path, "volume");
	read_obstacle_mesh(units, volume_obstacle, "", 2, vertices, codim_vertices, codim_edges, faces);
	CHECK(codim_edges.rows() == 1);
	CHECK(faces.size() == 0);
}

TEST_CASE("geometry reader obstacle geometry arrays and planes", "[geometry][geometry_reader]")
{
	using namespace polyfem;
	using namespace polyfem::mesh;

	const Units units;
	const std::filesystem::path line_path = write_geometry_reader_obj(
		"array_line.obj",
		"v 0 0 0\n"
		"v 1 0 0\n"
		"l 1 2\n");

	json line_array = obstacle_mesh_json(line_path, "edges");
	line_array["enabled"] = true;
	line_array["is_obstacle"] = true;
	line_array["type"] = "mesh_array";
	line_array["array"] = {{"relative", false}, {"offset", 2.0}, {"size", json::array({2, 1})}};

	Obstacle obstacle = read_obstacle_geometry(units, json::array({line_array}), {}, {}, "", 2);
	CHECK(obstacle.dim() == 2);
	CHECK(obstacle.n_vertices() == 4);
	CHECK(obstacle.n_edges() == 2);

	json plane = {
		{"enabled", true},
		{"is_obstacle", true},
		{"type", "plane"},
		{"point", json::array({0.0, 0.0})},
		{"normal", json::array({0.0, 1.0})}};
	json ground = {
		{"enabled", true},
		{"is_obstacle", true},
		{"type", "ground"},
		{"height", 2.0}};
	json disabled = {
		{"enabled", false},
		{"is_obstacle", true},
		{"type", "plane"},
		{"point", json::array({0.0, 0.0})},
		{"normal", json::array({1.0, 0.0})}};

	Obstacle planes = read_obstacle_geometry(units, json::array({disabled, plane, ground}), {}, {}, "", 2);
	REQUIRE(planes.planes().size() == 2);
	CHECK(planes.planes()[0].normal().norm() == Catch::Approx(1.0).margin(1e-12));
	CHECK(planes.planes()[1].normal().norm() == Catch::Approx(1.0).margin(1e-12));

	json invalid = {
		{"enabled", true},
		{"is_obstacle", true},
		{"type", "invalid"}};
	REQUIRE_THROWS(read_obstacle_geometry(units, json::array({invalid}), {}, {}, "", 2));
	REQUIRE_THROWS(read_fem_geometry(units, json::array(), ""));
}
