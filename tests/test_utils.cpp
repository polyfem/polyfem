////////////////////////////////////////////////////////////////////////////////
#include <polyfem/utils/InterpolatedFunction.hpp>
#include <polyfem/utils/RBFInterpolation.hpp>
#include <polyfem/utils/Bessel.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#ifdef POLYFEM_WITH_ITR
#include <wmtk/TriMesh.h>
#endif

#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <filesystem>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::mesh;
using namespace polyfem::io;
using namespace polyfem::utils;

namespace
{
	std::unique_ptr<Mesh> create_test_triangle_mesh()
	{
		Eigen::MatrixXd vertices(3, 2);
		vertices << 0, 0,
			2, 0,
			0, 1;

		Eigen::MatrixXi cells(1, 3);
		cells << 0, 1, 2;

		return Mesh::create(vertices, cells);
	}

	std::unique_ptr<Mesh> create_test_quad_mesh()
	{
		Eigen::MatrixXd vertices(4, 2);
		vertices << 0, 0,
			2, 0,
			2, 1,
			0, 1;

		Eigen::MatrixXi cells(1, 4);
		cells << 0, 1, 2, 3;

		return Mesh::create(vertices, cells);
	}

	std::unique_ptr<Mesh> create_test_tetra_mesh()
	{
		Eigen::MatrixXd vertices(4, 3);
		vertices << 0, 0, 0,
			1, 0, 0,
			0, 1, 0,
			0, 0, 1;

		Eigen::MatrixXi cells(1, 4);
		cells << 0, 1, 2, 3;

		return Mesh::create(vertices, cells);
	}

	std::unique_ptr<Mesh> create_test_hex_mesh()
	{
		Eigen::MatrixXd vertices(8, 3);
		vertices << 0, 0, 0,
			1, 0, 0,
			1, 1, 0,
			0, 1, 0,
			0, 0, 1,
			1, 0, 1,
			1, 1, 1,
			0, 1, 1;

		Eigen::MatrixXi cells(1, 8);
		cells << 0, 1, 2, 3, 4, 5, 6, 7;

		return Mesh::create(vertices, cells);
	}

	void require_sparse_equal(const StiffnessMatrix &actual, const Eigen::MatrixXd &expected)
	{
		const Eigen::MatrixXd dense(actual);
		REQUIRE(dense.rows() == expected.rows());
		REQUIRE(dense.cols() == expected.cols());
		REQUIRE((dense - expected).norm() == Catch::Approx(0).margin(1e-12));
	}
} // namespace

TEST_CASE("interpolated_fun_2d", "[utils]")
{
	Eigen::MatrixXd pts(3, 2);
	pts << 0, 0,
		3, 0,
		0, 3;

	Eigen::MatrixXi tri(1, 3);
	tri << 0, 1, 2;
	Eigen::MatrixXd fun(3, 4);
	fun.setRandom();

	Eigen::MatrixXd pt(1, 2);
	pt << 1, 1;

	InterpolatedFunction2d i_fun(fun, pts, tri);
	const auto res = i_fun.interpolate(pt);

	REQUIRE((fun.colwise().mean() - res).norm() == Catch::Approx(0).margin(1e-10));
}

TEST_CASE("rbf_interpolate", "[utils]")
{
#ifndef POLYFEM_OPENCL
	Eigen::MatrixXd in_pts(10, 3);
	in_pts.col(0) << 0.73142708, 0.15639157, 0.06799852, 0.61980247, 0.70461343, 0.96155237, 0.18068249, 0.09782913, 0.36740639, 0.26763186;
	in_pts.col(1) << 0.94896831, 0.37164925, 0.86693048, 0.87339727, 0.18393119, 0.19822407, 0.54455402, 0.98657281, 0.541773, 0.19644425;
	in_pts.col(2) << 0.41745562, 0.98444505, 0.15567433, 0.09762302, 0.69628704, 0.05620348, 0.34966505, 0.60069814, 0.79617982, 0.4012071;

	Eigen::MatrixXd fun(10, 1);
	fun << 0.14689884, 0.83814805, 0.67897605, 0.61621774, 0.12150901, 0.20614193, 0.27911847, 0.62222035, 0.98755679, 0.40910887;

	const double eps = 0.4052917899199118;

	RBFInterpolation rbf_fun(fun, in_pts, "multiquadric", eps);

	Eigen::MatrixXd out_pts(20, 3);
	const Eigen::MatrixXd t = VectorNd::LinSpaced(20, 0, 1);
	out_pts.col(0) = t;
	out_pts.col(1) = t;
	out_pts.col(2) = t;

	const auto actual = rbf_fun.interpolate(out_pts);

	Eigen::MatrixXd expected(20, 1);
	expected << 0.516894154053909, 0.476058965730433, 0.435748570197800, 0.397220357669309, 0.362961970820606, 0.337549413480876, 0.327784652734946, 0.339982127556460, 0.374173324760153, 0.420108170328222, 0.460205192638323, 0.478147060651189, 0.467001601598498, 0.431400523024562, 0.383076112995277, 0.334107457785502, 0.292629952299549, 0.262190022964549, 0.243125356328612, 0.234186308102585;

	for (int i = 0; i < 20; ++i)
		REQUIRE(actual(i) == Catch::Approx(expected(i)).margin(1e-10));
#endif
}

TEST_CASE("bessel", "[utils]")
{
	REQUIRE(bessy0(0.1) == Catch::Approx(-1.534238651350367).margin(1e-8));
	REQUIRE(bessy0(1.) == Catch::Approx(0.088256964215677).margin(1e-8));
	REQUIRE(bessy0(10.) == Catch::Approx(0.055671167283599).margin(1e-8));
	REQUIRE(bessy0(100.) == Catch::Approx(-0.077244313365083).margin(1e-8));

	REQUIRE(bessy1(0.1) == Catch::Approx(-6.458951094702027).margin(1e-8));
	REQUIRE(bessy1(1.) == Catch::Approx(-0.781212821300289).margin(1e-8));
	REQUIRE(bessy1(10.) == Catch::Approx(0.249015424206954).margin(1e-8));
	REQUIRE(bessy1(100.) == Catch::Approx(-0.020372312002760).margin(1e-8));
}

TEST_CASE("expression", "[utils]")
{
	json jexpr = {{"value", "x^2+sqrt(x*y)+sin(z)*x"}};
	json jexpr2d = {{"value", "x^2+sqrt(x*y)"}};
	json jval = {{"value", 1}};

	utils::ExpressionValue expr;
	expr.init(jexpr["value"], "");
	utils::ExpressionValue expr2d;
	expr2d.init(jexpr2d["value"], "");
	utils::ExpressionValue val;
	val.init(jval["value"], "");

	expr.set_unit_type("");
	expr2d.set_unit_type("");
	val.set_unit_type("");

	REQUIRE(expr(2, 3, 4) == Catch::Approx(2. * 2. + sqrt(2. * 3.) + sin(4.) * 2.).margin(1e-10));
	REQUIRE(expr2d(2, 3) == Catch::Approx(2. * 2. + sqrt(2. * 3.)).margin(1e-10));
	REQUIRE(val(2, 3, 4) == Catch::Approx(1).margin(1e-16));

	utils::ExpressionValue time_expr;
	time_expr.init(json::array({"x", "2*x"}), "");
	time_expr.set_t(json::array({0, 1}));
	time_expr.set_unit_type("");

	REQUIRE_FALSE(time_expr.is_zero());
	REQUIRE(time_expr(3, 0, 0, 0) == Catch::Approx(3).margin(1e-16));
	REQUIRE(time_expr(3, 0, 0, 1) == Catch::Approx(6).margin(1e-16));

#ifdef POLYFEM_WITH_PYTHON
	const std::filesystem::path python_file = std::filesystem::temp_directory_path() / "polyfem_python_expression_test.py";
	{
		std::ofstream file(python_file);
		REQUIRE(file.is_open());
		file << "def value(x, y, z, t, index):\n"
			 << "    return x + 2 * y + 3 * z + 4 * t + index\n";
	}

	utils::ExpressionValue python_expr;
	python_expr.init({{"file_name", python_file.string()}, {"function_name", "value"}}, "");
	python_expr.set_unit_type("");

	REQUIRE_FALSE(python_expr.is_zero());
	REQUIRE(python_expr(1, 2, 3, 4, 5) == Catch::Approx(35).margin(1e-16));

	std::filesystem::remove(python_file);
#endif
}

TEST_CASE("mshreader", "[utils]")
{
	const std::string path = POLYFEM_DATA_DIR;
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi cells;
	const auto mesh = Mesh::create(path + "/circle2.msh");
	REQUIRE(mesh);
}

TEST_CASE("inverse", "[utils]")
{
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat = Eigen::MatrixXd::Random(1, 1);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat2 = Eigen::MatrixXd::Random(2, 2);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat3 = Eigen::MatrixXd::Random(3, 3);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat_inv = mat.inverse();
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat2_inv = mat2.inverse();
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat3_inv = mat3.inverse();

	REQUIRE(((utils::inverse(mat) - mat_inv)).norm() == Catch::Approx(0).margin(1e-12));
	REQUIRE(((utils::inverse(mat2) - mat2_inv)).norm() == Catch::Approx(0).margin(1e-12));
	REQUIRE(((utils::inverse(mat3) - mat3_inv)).norm() == Catch::Approx(0).margin(1e-12));
}

TEST_CASE("matrix utils row-major transforms and sparse mappings", "[utils][matrix]")
{
	Eigen::MatrixXd X(2, 3);
	X << 1, 2, 3,
		4, 5, 6;
	const Eigen::VectorXd x = flatten(X);
	REQUIRE(x.transpose().isApprox(Eigen::RowVectorXd::LinSpaced(6, 1, 6)));
	REQUIRE(unflatten(x, 3).isApprox(X));
	REQUIRE(unflatten(Eigen::VectorXd(), 3).rows() == 0);
	REQUIRE(unflatten(Eigen::VectorXd(), 3).cols() == 3);

	Eigen::MatrixXd mat;
	Eigen::Vector4d vec4;
	vec4 << 1, 2, 3, 4;
	vector2matrix(vec4, mat);
	Eigen::Matrix2d mat_expected;
	mat_expected << 1, 2,
		3, 4;
	REQUIRE(mat.isApprox(mat_expected));
	Eigen::VectorXd vec9 = Eigen::VectorXd::LinSpaced(9, 1, 9);
	vector2matrix(vec9, mat);
	REQUIRE(mat.isApprox(unflatten(vec9, 3)));
	REQUIRE_THROWS(vector2matrix(Eigen::Vector3d::Ones(), mat));

	show_matrix_stats(Eigen::Matrix2d::Identity());

	Eigen::SparseMatrix<double> sparse(3, 3);
	const std::vector<Eigen::Triplet<double>> sparse_entries = {
		Eigen::Triplet<double>(0, 1, 2),
		Eigen::Triplet<double>(0, 2, 3),
		Eigen::Triplet<double>(2, 1, 4),
	};
	sparse.setFromTriplets(sparse_entries.begin(), sparse_entries.end());

	Eigen::MatrixXd lumped_expected = Eigen::MatrixXd::Zero(3, 3);
	lumped_expected(0, 0) = 5;
	lumped_expected(2, 2) = 4;
	require_sparse_equal(lump_matrix(sparse), lumped_expected);

	Eigen::MatrixXd full_dense(4, 4);
	full_dense << 1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16;
	StiffnessMatrix full = full_dense.sparseView();
	StiffnessMatrix reduced;
	full_to_reduced_matrix(4, 2, {1, 3}, full, reduced);
	Eigen::MatrixXd reduced_expected(2, 2);
	reduced_expected << 1, 3,
		9, 11;
	require_sparse_equal(reduced, reduced_expected);

	full_to_reduced_matrix(4, 4, {}, full, reduced);
	require_sparse_equal(reduced, full_dense);
}

TEST_CASE("matrix utils reorder and scatter block data", "[utils][matrix]")
{
	Eigen::MatrixXd in(4, 2);
	in << 1, 2,
		3, 4,
		5, 6,
		7, 8;
	Eigen::Vector2i in_to_out;
	in_to_out << 1, 0;
	const Eigen::MatrixXd out = reorder_matrix(in, in_to_out, -1, 2);
	REQUIRE(out.topRows(2).isApprox(in.bottomRows(2)));
	REQUIRE(out.bottomRows(2).isApprox(in.topRows(2)));
	REQUIRE(unreorder_matrix(out, in_to_out, -1, 2).isApprox(in));

	Eigen::Vector2i partial_map;
	partial_map << 2, -1;
	const Eigen::MatrixXd partial = reorder_matrix(in, partial_map, 3, 2);
	REQUIRE(partial.middleRows(4, 2).isApprox(in.topRows(2)));
	CHECK(std::isnan(partial(0, 0)));

	Eigen::MatrixXi indices(2, 2);
	indices << 0, 2,
		1, 3;
	Eigen::Vector4i index_mapping;
	index_mapping << 3, 2, 1, 0;
	Eigen::MatrixXi mapped_expected(2, 2);
	mapped_expected << 3, 1,
		2, 0;
	REQUIRE(map_index_matrix(indices, index_mapping).isApprox(mapped_expected));

	Eigen::MatrixXd appended(0, 0);
	Eigen::MatrixXd rows(2, 2);
	rows << 1, 2,
		3, 4;
	append_rows(appended, rows);
	append_rows_of_zeros(appended, 1);
	REQUIRE(appended.topRows(2).isApprox(rows));
	REQUIRE(appended.bottomRows(1).isZero());

	Eigen::MatrixXd A(2, 2);
	A << 1, 2,
		0, 3;
	Eigen::MatrixXd b(2, 2);
	b << 10, 20,
		30, 40;
	StiffnessMatrix Aout;
	Eigen::MatrixXd bout;
	scatter_matrix(6, 2, A, b, {2, 0}, Aout, bout);

	Eigen::MatrixXd scatter_expected = Eigen::MatrixXd::Zero(4, 6);
	scatter_expected(0, 4) = 1;
	scatter_expected(1, 5) = 1;
	scatter_expected(0, 0) = 2;
	scatter_expected(1, 1) = 2;
	scatter_expected(2, 0) = 3;
	scatter_expected(3, 1) = 3;
	require_sparse_equal(Aout, scatter_expected);
	REQUIRE(bout.isApprox(Eigen::Vector4d(10, 20, 30, 40)));

	Eigen::MatrixXd b_scalar(2, 1);
	b_scalar << 100, 200;
	Eigen::MatrixXd A_scalar(2, 4);
	A_scalar << 1, 2, 0, 3,
		0, 4, 5, 0;
	scatter_matrix(6, 2, A_scalar, b_scalar, {2, 0}, Aout, bout);
	Eigen::MatrixXd scalar_scatter_expected = Eigen::MatrixXd::Zero(2, 6);
	scalar_scatter_expected(0, 4) = 1;
	scalar_scatter_expected(0, 5) = 2;
	scalar_scatter_expected(0, 1) = 3;
	scalar_scatter_expected(1, 5) = 4;
	scalar_scatter_expected(1, 0) = 5;
	require_sparse_equal(Aout, scalar_scatter_expected);
	REQUIRE(bout.isApprox(b_scalar));

	REQUIRE_THROWS(scatter_matrix_col(4, 2, A, b, {1, 0}, Aout, bout));

	scatter_matrix(4, 2, {2, 2}, {0, 0, 1}, {0, 1, 1}, {1, 2, 0}, b, {1, 0}, Aout, bout);
	Eigen::MatrixXd sparse_scatter_expected = Eigen::MatrixXd::Zero(4, 4);
	sparse_scatter_expected(0, 2) = 1;
	sparse_scatter_expected(1, 3) = 1;
	sparse_scatter_expected(0, 0) = 2;
	sparse_scatter_expected(1, 1) = 2;
	require_sparse_equal(Aout, sparse_scatter_expected);
	REQUIRE(bout.isApprox(Eigen::Vector4d(10, 20, 30, 40)));

	scatter_matrix(4, 2, {2, 4}, {0, 1}, {0, 3}, {5, 6}, b_scalar, {1, 0}, Aout, bout);
	Eigen::MatrixXd sparse_scalar_expected = Eigen::MatrixXd::Zero(2, 4);
	sparse_scalar_expected(0, 2) = 5;
	sparse_scalar_expected(1, 1) = 6;
	require_sparse_equal(Aout, sparse_scalar_expected);
	REQUIRE(bout.isApprox(b_scalar));

	scatter_matrix_col(4, 2, {2, 2}, {0, 1}, {1, 0}, {2, 3}, b, {1, 0}, Aout, bout);
	Eigen::MatrixXd scatter_col_expected = Eigen::MatrixXd::Zero(4, 4);
	scatter_col_expected(2, 2) = 2;
	scatter_col_expected(3, 3) = 2;
	scatter_col_expected(0, 0) = 3;
	scatter_col_expected(1, 1) = 3;
	require_sparse_equal(Aout, scatter_col_expected);
	REQUIRE(bout.isApprox(Eigen::Vector4d(30, 40, 10, 20)));

	Eigen::MatrixXd b_col_scalar(4, 1);
	b_col_scalar << 10, 20, 30, 40;
	scatter_matrix_col(4, 2, {4, 2}, {0, 3}, {1, 0}, {7, 8}, b_col_scalar, {1, 0}, Aout, bout);
	Eigen::MatrixXd scatter_col_scalar_expected = Eigen::MatrixXd::Zero(4, 2);
	scatter_col_scalar_expected(2, 1) = 7;
	scatter_col_scalar_expected(1, 0) = 8;
	require_sparse_equal(Aout, scatter_col_scalar_expected);
	REQUIRE(bout.isApprox(Eigen::Vector4d(30, 40, 10, 20)));
}

TEST_CASE("boundary sampler local coordinates and triangle-edge quadrature", "[utils][boundary_sampler]")
{
	const Eigen::Matrix2d tri_edge = BoundarySampler::tri_local_node_coordinates_from_edge(0);
	REQUIRE(tri_edge.rows() == 2);
	REQUIRE(tri_edge.cols() == 2);
	REQUIRE((tri_edge.array() >= 0).all());
	REQUIRE((tri_edge.array() <= 1).all());

	const Eigen::MatrixXd tet_face = BoundarySampler::tet_local_node_coordinates_from_face(0);
	REQUIRE(tet_face.rows() == 3);
	REQUIRE(tet_face.cols() == 3);
	for (int r = 0; r < tet_face.rows(); ++r)
		CHECK(tet_face.row(r).sum() <= 1 + 1e-12);

	CHECK(BoundarySampler::prism_local_node_coordinates_from_face(0).rows() == 3);
	CHECK(BoundarySampler::prism_local_node_coordinates_from_face(2).rows() == 4);
	CHECK(BoundarySampler::pyramid_local_node_coordinates_from_face(0).rows() == 4);
	CHECK(BoundarySampler::pyramid_local_node_coordinates_from_face(1).rows() == 3);
	CHECK(BoundarySampler::hex_local_node_coordinates_from_face(0).rows() == 4);

	Eigen::MatrixXd normal;
	BoundarySampler::normal_for_tri_edge(0, normal);
	REQUIRE(normal.rows() == 1);
	REQUIRE(normal.cols() == 2);
	REQUIRE(normal.norm() == Catch::Approx(1).margin(1e-12));
	BoundarySampler::normal_for_tri_face(0, normal);
	REQUIRE(normal.rows() == 1);
	REQUIRE(normal.cols() == 3);
	REQUIRE(normal.norm() == Catch::Approx(1).margin(1e-12));

	Eigen::MatrixXd uv, samples;
	BoundarySampler::sample_parametric_tri_edge(0, 3, uv, samples);
	REQUIRE(uv.rows() == 3);
	REQUIRE(samples.rows() == 3);
	REQUIRE(samples.topRows(1).isApprox(tri_edge.topRows(1)));
	REQUIRE(samples.bottomRows(1).isApprox(tri_edge.bottomRows(1)));

	const auto mesh = create_test_triangle_mesh();
	Eigen::MatrixXd points;
	Eigen::VectorXd weights;
	BoundarySampler::quadrature_for_tri_edge(0, 2, 0, *mesh, uv, points, weights);
	REQUIRE(uv.rows() == weights.size());
	REQUIRE(points.rows() == weights.size());
	REQUIRE(weights.sum() == Catch::Approx(mesh->edge_length(0)).margin(1e-12));

	LocalBoundary local_boundary(0, BoundaryType::TRI_LINE);
	local_boundary.add_boundary_primitive(0, 0);
	Eigen::VectorXi global_ids;
	BoundarySampler::sample_boundary(local_boundary, 3, *mesh, false, uv, samples, global_ids);
	REQUIRE(samples.rows() == 3);
	REQUIRE(global_ids.size() == 3);
	CHECK((global_ids.array() == 0).all());

	Eigen::MatrixXd normals;
	BoundarySampler::boundary_quadrature(local_boundary, {{2, 0}}, *mesh, false, uv, points, normals, weights, global_ids);
	REQUIRE(points.rows() == weights.size());
	REQUIRE(normals.rows() == weights.size());
	REQUIRE(global_ids.size() == weights.size());
	CHECK((global_ids.array() == 0).all());
	CHECK(weights.sum() == Catch::Approx(mesh->edge_length(0)).margin(1e-12));
}

TEST_CASE("boundary sampler quad and volume face sampling", "[utils][boundary_sampler]")
{
	Eigen::MatrixXd uv, samples;
	BoundarySampler::sample_parametric_quad_edge(1, 4, uv, samples);
	REQUIRE(uv.rows() == 4);
	REQUIRE(uv.cols() == 2);
	REQUIRE(samples.rows() == 4);
	REQUIRE(samples.cols() == 2);
	for (int i = 0; i < uv.rows(); ++i)
		CHECK(uv.row(i).sum() == Catch::Approx(1.0).margin(1e-12));
	CHECK(samples.topRows(1).isApprox(BoundarySampler::quad_local_node_coordinates_from_edge(1).topRows(1)));
	CHECK(samples.bottomRows(1).isApprox(BoundarySampler::quad_local_node_coordinates_from_edge(1).bottomRows(1)));

	BoundarySampler::sample_parametric_quad_face(0, 3, uv, samples);
	REQUIRE(samples.rows() == 9);
	REQUIRE(samples.cols() == 3);
	CHECK((samples.array() >= -1e-12).all());
	CHECK((samples.array() <= 1 + 1e-12).all());

	BoundarySampler::sample_parametric_tri_face(2, 4, uv, samples);
	REQUIRE(samples.cols() == 3);
	REQUIRE(uv.cols() == 3);
	REQUIRE(samples.rows() == uv.rows());
	for (int i = 0; i < uv.rows(); ++i)
		CHECK(uv.row(i).sum() == Catch::Approx(1.0).margin(1e-12));

	BoundarySampler::sample_parametric_prism_face(0, 4, uv, samples);
	REQUIRE(samples.cols() == 3);
	REQUIRE(uv.cols() == 3);
	for (int i = 0; i < uv.rows(); ++i)
		CHECK(uv.row(i).sum() == Catch::Approx(1.0).margin(1e-12));

	BoundarySampler::sample_parametric_prism_face(3, 3, uv, samples);
	REQUIRE(samples.rows() == 9);
	REQUIRE(samples.cols() == 3);

	BoundarySampler::sample_parametric_pyramid_face(0, 3, uv, samples);
	REQUIRE(samples.rows() == 9);
	REQUIRE(samples.cols() == 3);

	BoundarySampler::sample_parametric_pyramid_face(2, 4, uv, samples);
	REQUIRE(samples.cols() == 3);
	REQUIRE(uv.cols() == 3);
	for (int i = 0; i < uv.rows(); ++i)
		CHECK(uv.row(i).sum() == Catch::Approx(1.0).margin(1e-12));

	Eigen::MatrixXd normal;
	for (int face = 0; face < 5; ++face)
	{
		BoundarySampler::normal_for_prism_face(face, normal);
		REQUIRE(normal.rows() == 1);
		REQUIRE(normal.cols() == 3);
		CHECK(normal.norm() == Catch::Approx(1.0).margin(1e-12));

		BoundarySampler::normal_for_pyramid_face(face, normal);
		REQUIRE(normal.rows() == 1);
		REQUIRE(normal.cols() == 3);
		CHECK(normal.norm() == Catch::Approx(1.0).margin(1e-12));
	}
}

TEST_CASE("boundary sampler quadrature dispatch for quads and 3d faces", "[utils][boundary_sampler]")
{
	const auto quad_mesh = create_test_quad_mesh();
	const auto tet_mesh = create_test_tetra_mesh();
	const auto hex_mesh = create_test_hex_mesh();

	Eigen::MatrixXd uv, points, normals;
	Eigen::VectorXd weights;
	Eigen::VectorXi global_ids;

	BoundarySampler::quadrature_for_quad_edge(0, 2, 0, *quad_mesh, uv, points, weights);
	REQUIRE(points.rows() == weights.size());
	REQUIRE(points.cols() == 2);
	CHECK(weights.sum() == Catch::Approx(quad_mesh->edge_length(0)).margin(1e-12));

	BoundarySampler::quadrature_for_tri_face(0, 2, 0, *tet_mesh, uv, points, weights);
	REQUIRE(points.rows() == weights.size());
	REQUIRE(points.cols() == 3);
	CHECK(weights.sum() == Catch::Approx(tet_mesh->tri_area(0)).margin(1e-12));

	BoundarySampler::quadrature_for_quad_face(0, 2, 0, *hex_mesh, uv, points, weights);
	REQUIRE(points.rows() == weights.size());
	REQUIRE(points.cols() == 3);
	CHECK(weights.sum() == Catch::Approx(hex_mesh->quad_area(0)).margin(1e-12));

	BoundarySampler::quadrature_for_prism_face(0, 2, 2, 0, *tet_mesh, uv, points, weights);
	CHECK(weights.sum() == Catch::Approx(tet_mesh->tri_area(0)).margin(1e-12));
	BoundarySampler::quadrature_for_prism_face(3, 2, 2, 0, *hex_mesh, uv, points, weights);
	CHECK(weights.sum() == Catch::Approx(hex_mesh->quad_area(0)).margin(1e-12));

	BoundarySampler::quadrature_for_pyramid_face(2, 2, 0, *tet_mesh, uv, points, weights);
	CHECK(weights.sum() == Catch::Approx(tet_mesh->tri_area(0)).margin(1e-12));
	BoundarySampler::quadrature_for_pyramid_face(0, 2, 0, *hex_mesh, uv, points, weights);
	CHECK(weights.sum() == Catch::Approx(hex_mesh->quad_area(0)).margin(1e-12));

	LocalBoundary quad_boundary(0, BoundaryType::QUAD_LINE);
	quad_boundary.add_boundary_primitive(0, 0);
	BoundarySampler::sample_boundary(quad_boundary, 3, *quad_mesh, false, uv, points, global_ids);
	REQUIRE(points.rows() == 3);
	CHECK((global_ids.array() == 0).all());

	BoundarySampler::boundary_quadrature(quad_boundary, {{2, 0}}, *quad_mesh, false, uv, points, normals, weights, global_ids);
	REQUIRE(points.rows() == weights.size());
	REQUIRE(normals.rows() == weights.size());
	CHECK(weights.sum() == Catch::Approx(quad_mesh->edge_length(0)).margin(1e-12));

	LocalBoundary tri_face_boundary(0, BoundaryType::TRI);
	tri_face_boundary.add_boundary_primitive(0, 0);
	BoundarySampler::boundary_quadrature(tri_face_boundary, {{2, 0}}, *tet_mesh, 0, false, uv, points, normals, weights);
	REQUIRE(points.rows() == weights.size());
	REQUIRE(normals.rows() == weights.size());
	CHECK(weights.sum() == Catch::Approx(tet_mesh->tri_area(0)).margin(1e-12));

	LocalBoundary quad_face_boundary(0, BoundaryType::QUAD);
	quad_face_boundary.add_boundary_primitive(0, 0);
	BoundarySampler::boundary_quadrature(quad_face_boundary, {{2, 0}}, *hex_mesh, 0, false, uv, points, normals, weights);
	REQUIRE(points.rows() == weights.size());
	REQUIRE(normals.rows() == weights.size());
	CHECK(weights.sum() == Catch::Approx(hex_mesh->quad_area(0)).margin(1e-12));
}

#ifdef POLYFEM_WITH_ITR
TEST_CASE("wmtk_instantiation", "[utils]")
{
	wmtk::TriMesh mesh;
}
#endif
