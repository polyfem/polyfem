#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/lagrangian/MatrixLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/PeriodicBoundaryLagrangianForm.hpp>
#include <polyfem/utils/Logger.hpp>

#include <catch2/catch_test_macros.hpp>
#include <polysolve/linear/Solver.hpp>

#include <cmath>
#include <map>
#include <memory>

using namespace polyfem;

namespace
{
	struct SquareSpace
	{
		std::unique_ptr<mesh::Mesh> mesh;
		std::vector<basis::ElementBases> bases;
		std::vector<mesh::LocalBoundary> local_boundary;
		int n_bases = 0;
	};

	SquareSpace make_square_space(const int left_boundary_id = 1)
	{
		Eigen::MatrixXd vertices(4, 2);
		vertices << 0, 0,
			1, 0,
			1, 1,
			0, 1;
		Eigen::MatrixXi cells(1, 4);
		cells << 0, 1, 2, 3;

		SquareSpace space;
		space.mesh = mesh::Mesh::create(vertices, cells);
		space.mesh->compute_boundary_ids([left_boundary_id](const size_t, const std::vector<int> &, const RowVectorNd &point, const bool is_boundary) {
			if (!is_boundary)
				return 0;
			if (std::abs(point.x()) < 1e-12)
				return left_boundary_id;
			if (std::abs(point.x() - 1) < 1e-12)
				return 2;
			if (std::abs(point.y()) < 1e-12)
				return 3;
			return 4;
		});

		std::map<int, basis::InterfaceData> poly_edge_to_data;
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;
		space.n_bases = basis::LagrangeBasis2d::build_bases(
			dynamic_cast<const mesh::Mesh2D &>(*space.mesh), "Laplacian",
			-1, -1, 2, false, false, false, false, false,
			space.bases, space.local_boundary, poly_edge_to_data, mesh_nodes);
		return space;
	}

	RowVectorNd point(const double x, const double y)
	{
		RowVectorNd result(2);
		result << x, y;
		return result;
	}

	void replace_node_with_corner_average(
		std::vector<basis::ElementBases> &bases,
		const RowVectorNd &node,
		const RowVectorNd &corner0,
		const RowVectorNd &corner1)
	{
		basis::Basis *target = nullptr;
		basis::Local2Global first, second;
		for (basis::ElementBases &element : bases)
		{
			for (basis::Basis &basis : element.bases)
			{
				for (const basis::Local2Global &global : basis.global())
				{
					if ((global.node - node).norm() < 1e-12)
						target = &basis;
					if ((global.node - corner0).norm() < 1e-12)
						first = global;
					if ((global.node - corner1).norm() < 1e-12)
						second = global;
				}
			}
		}
		REQUIRE(target != nullptr);
		REQUIRE(first.index >= 0);
		REQUIRE(second.index >= 0);
		target->global() = {
			basis::Local2Global(first.index, first.node, 0.5),
			basis::Local2Global(second.index, second.node, 0.5)};
	}

	void move_global_node(
		std::vector<basis::ElementBases> &bases,
		const RowVectorNd &from,
		const RowVectorNd &to)
	{
		int global_index = -1;
		for (const basis::ElementBases &element : bases)
		{
			for (const basis::Basis &basis : element.bases)
			{
				for (const basis::Local2Global &global : basis.global())
				{
					if ((global.node - from).norm() < 1e-12)
						global_index = global.index;
				}
			}
		}
		REQUIRE(global_index >= 0);

		for (basis::ElementBases &element : bases)
		{
			for (basis::Basis &basis : element.bases)
			{
				for (basis::Local2Global &global : basis.global())
				{
					if (global.index == global_index)
						global.node = to;
				}
			}
		}
	}
} // namespace

TEST_CASE("periodic boundary form builds equality constraints", "[periodic][constraints]")
{
	const SquareSpace space = make_square_space();
	const int value_dim = 2;
	const solver::PeriodicBoundaryLagrangianForm form(
		space.n_bases * value_dim, value_dim, *space.mesh,
		space.bases, space.local_boundary, {{1, 2}}, 1e-5);

	const StiffnessMatrix &A = form.constraint_matrix();
	CHECK(A.rows() == 3 * value_dim);
	CHECK(A.cols() == space.n_bases * value_dim);
	CHECK(A.nonZeros() == 2 * A.rows());
	CHECK(form.constraint_value().isZero());
	CHECK((A * Eigen::VectorXd::Ones(A.cols())).isZero());

	Eigen::VectorXd positive = Eigen::VectorXd::Zero(A.rows());
	Eigen::VectorXd negative = Eigen::VectorXd::Zero(A.rows());
	for (int outer = 0; outer < A.outerSize(); ++outer)
	{
		for (StiffnessMatrix::InnerIterator it(A, outer); it; ++it)
		{
			positive(it.row()) += it.value() > 0 ? it.value() : 0;
			negative(it.row()) += it.value() < 0 ? it.value() : 0;
		}
	}
	CHECK(positive.isOnes());
	CHECK((-negative).isOnes());
}

TEST_CASE("periodic boundary form accepts boundary ID zero", "[periodic][constraints]")
{
	const SquareSpace space = make_square_space(0);
	const solver::PeriodicBoundaryLagrangianForm form(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, {{0, 2}}, 1e-5);

	CHECK(form.constraint_matrix().rows() == 3);
	CHECK(form.constraint_value().isZero());
}

TEST_CASE("periodic boundary matching skips already paired DoFs", "[periodic][constraints]")
{
	SquareSpace space = make_square_space();
	move_global_node(space.bases, point(0, 0), point(0, 0.25));
	move_global_node(space.bases, point(0, 0.5), point(0, 0.35));
	move_global_node(space.bases, point(1, 0.5), point(1, 0.6));

	const solver::PeriodicBoundaryLagrangianForm form(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, {{1, 2}}, 1);

	CHECK(form.constraint_matrix().rows() == 3);
	CHECK(form.constraint_value().isZero());
}

TEST_CASE("periodic boundary form rejects a missing pair", "[periodic][constraints]")
{
	const SquareSpace space = make_square_space();
	REQUIRE_THROWS(solver::PeriodicBoundaryLagrangianForm(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, {{1, 99}}, 1e-5));
}

TEST_CASE("periodic boundary form rejects invalid pair settings", "[periodic][constraints]")
{
	const SquareSpace space = make_square_space();
	REQUIRE_THROWS(solver::PeriodicBoundaryLagrangianForm(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, {{1, 1}}, 1e-5));
	REQUIRE_THROWS(solver::PeriodicBoundaryLagrangianForm(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, {{1, 2}}, 0));
}

TEST_CASE("periodic boundary form keeps weighted trace mappings", "[periodic][constraints]")
{
	SquareSpace space = make_square_space();
	replace_node_with_corner_average(space.bases, point(0, 0.5), point(0, 0), point(0, 1));
	replace_node_with_corner_average(space.bases, point(1, 0.5), point(1, 0), point(1, 1));

	const solver::PeriodicBoundaryLagrangianForm form(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, {{1, 2}}, 1e-5);
	const Eigen::MatrixXd A(form.constraint_matrix());

	CHECK(A.rows() == 3);
	CHECK(form.constraint_value().isZero());
	CHECK((A * Eigen::VectorXd::Ones(A.cols())).isZero());
	bool found_weighted_row = false;
	for (int row = 0; row < A.rows(); ++row)
		found_weighted_row |= (A.row(row).array() != 0).count() == 4;
	CHECK(found_weighted_row);
}

TEST_CASE("periodic boundary form rejects boundaries without a translated match", "[periodic][constraints]")
{
	const SquareSpace space = make_square_space();
	REQUIRE_THROWS(solver::PeriodicBoundaryLagrangianForm(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, {{1, 3}}, 1e-5));
}

TEST_CASE("periodic and Dirichlet forms share the existing QR reduction", "[periodic][constraints]")
{
	const SquareSpace space = make_square_space();
	auto periodic = std::make_shared<solver::PeriodicBoundaryLagrangianForm>(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, std::array<int, 2>{{1, 2}}, 1e-5);

	StiffnessMatrix dbc_A(1, space.n_bases);
	dbc_A.insert(0, 0) = 1;
	Eigen::MatrixXd dbc_b(1, 1);
	dbc_b(0) = 2;
	auto dbc = std::make_shared<solver::MatrixLagrangianForm>(dbc_A, dbc_b);

	std::vector<std::shared_ptr<solver::AugmentedLagrangianForm>> constraints{dbc, periodic};
	std::shared_ptr<polysolve::linear::Solver> linear_solver = polysolve::linear::Solver::create(
		json({{"solver", "Eigen::SparseLU"}}), logger());
	StiffnessMatrix mass(space.n_bases, space.n_bases);
	mass.setIdentity();
	solver::NLProblem problem(
		space.n_bases, 0, {}, constraints, linear_solver,
		1, 1, mass, 1);

	const Eigen::VectorXd full = problem.reduced_to_full(Eigen::VectorXd::Zero(problem.reduced_size()));
	CHECK((dbc->constraint_matrix() * full - dbc->constraint_value()).norm() < 1e-10);
	CHECK((periodic->constraint_matrix() * full).norm() < 1e-10);
}

TEST_CASE("two periodic directions leave redundant corner constraints to QR", "[periodic][constraints]")
{
	const SquareSpace space = make_square_space();
	auto horizontal = std::make_shared<solver::PeriodicBoundaryLagrangianForm>(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, std::array<int, 2>{{1, 2}}, 1e-5);
	auto vertical = std::make_shared<solver::PeriodicBoundaryLagrangianForm>(
		space.n_bases, 1, *space.mesh, space.bases,
		space.local_boundary, std::array<int, 2>{{3, 4}}, 1e-5);

	std::vector<std::shared_ptr<solver::AugmentedLagrangianForm>> constraints{horizontal, vertical};
	std::shared_ptr<polysolve::linear::Solver> linear_solver = polysolve::linear::Solver::create(
		json({{"solver", "Eigen::SparseLU"}}), logger());
	StiffnessMatrix mass(space.n_bases, space.n_bases);
	mass.setIdentity();
	solver::NLProblem problem(
		space.n_bases, 0, {}, constraints, linear_solver,
		1, 1, mass, 1);

	CHECK(problem.reduced_size() == space.n_bases - 5);
	const Eigen::VectorXd full = problem.reduced_to_full(Eigen::VectorXd::Zero(problem.reduced_size()));
	CHECK((horizontal->constraint_matrix() * full).norm() < 1e-10);
	CHECK((vertical->constraint_matrix() * full).norm() < 1e-10);
}
