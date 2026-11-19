#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>
#include <polyfem/assembler/MacroStrain.hpp>
#include <polyfem/solver/forms/lagrangian/MatrixLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/PeriodicBoundaryLagrangianForm.hpp>
#include <polyfem/utils/Logger.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
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
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;
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
		space.mesh = mesh::Mesh::create(mesh::MeshData(vertices, cells));
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
		space.n_bases = basis::LagrangeBasis2d::build_bases(
			dynamic_cast<const mesh::Mesh2D &>(*space.mesh), "Laplacian",
			-1, -1, 2, false, false, false, false, false,
			space.bases, space.local_boundary, poly_edge_to_data, space.mesh_nodes);
		return space;
	}

	class CallbackForm : public solver::Form
	{
	public:
		mutable Eigen::VectorXd last_x, last_x0;
		std::string name() const override { return "callback test"; }
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			last_x0 = x0;
			last_x = x1;
			return true;
		}
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			return is_step_valid(x0, x1);
		}
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			is_step_valid(x0, x1);
			return 1;
		}
		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override { is_step_valid(x0, x1); }
		void post_step(const polysolve::nonlinear::PostStepData &data) override { last_x = data.x; }
		void solution_changed(const Eigen::VectorXd &x) override { last_x = x; }
		void init_lagging(const Eigen::VectorXd &x) override { last_x = x; }
		void update_lagging(const Eigen::VectorXd &x, const int) override { last_x = x; }
		void update_quantities(const double, const Eigen::VectorXd &x) override { last_x = x; }

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override { return x.squaredNorm() / 2; }
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradient) const override { gradient = x; }
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override
		{
			hessian.resize(x.size(), x.size());
			hessian.setIdentity();
		}
	};

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

TEST_CASE("homogenization callbacks include macro displacement", "[periodic][macro_strain]")
{
	const bool symmetric = GENERATE(false, true);
	CAPTURE(symmetric);
	const SquareSpace space = make_square_space();
	const int ndof = 2 * space.n_bases;
	assembler::MacroStrainValue macro_strain;
	macro_strain.init(2, {{"value", {{-0.2, 0}, {0, -0.15}}}, {"fixed_components", {0, 3}}}, io::FileSystemIO("."));
	auto form = std::make_shared<CallbackForm>();
	auto extended_form = std::make_shared<CallbackForm>();
	auto periodic = std::make_shared<solver::PeriodicBoundaryLagrangianForm>(
		ndof, 2, *space.mesh, space.bases, space.local_boundary,
		std::array<int, 2>{{1, 2}}, 1e-5);
	std::shared_ptr<polysolve::linear::Solver> linear_solver = polysolve::linear::Solver::create(
		json({{"solver", "Eigen::SparseLU"}}), logger());
	StiffnessMatrix mass(ndof, ndof);
	mass.setIdentity();
	solver::NLHomoProblem problem(
		ndof, macro_strain, space.n_bases, space.mesh_nodes, 0,
		{form}, {periodic}, symmetric, linear_solver, 1, 1, mass, 2);
	problem.add_form(extended_form);

	for (const bool fixed : {false, true})
	{
		CAPTURE(fixed);
		problem.set_fixed_entry(fixed ? macro_strain.get_fixed_entry() : Eigen::VectorXi());
		Eigen::VectorXd extended = Eigen::VectorXd::Zero(ndof + 4);
		extended.tail(4) << -0.2, 0.03, 0.03, -0.15;
		const Eigen::VectorXd x = problem.extended_to_reduced(extended);
		const Eigen::VectorXd x0 = Eigen::VectorXd::Zero(x.size());
		const Eigen::VectorXd full = problem.reduced_to_full(x);
		const Eigen::VectorXd full0 = problem.reduced_to_full(x0);
		REQUIRE(full.norm() > 0.1);
		const auto check_callback = [&]() {
			CHECK(form->last_x.isApprox(full, 1e-12));
			CHECK(extended_form->last_x.isApprox(extended, 1e-12));
		};
		CHECK(problem.is_step_valid(x0, x));
		check_callback();
		CHECK(form->last_x0.isApprox(full0, 1e-12));
		CHECK(problem.is_step_collision_free(x0, x));
		check_callback();
		CHECK(problem.max_step_size(x0, x) == 1);
		check_callback();
		problem.line_search_begin(x0, x);
		check_callback();
		problem.line_search_end();
		problem.solution_changed(x);
		check_callback();
		problem.init_lagging(x);
		check_callback();
		problem.update_lagging(x, 1);
		check_callback();
		problem.update_quantities(0, x);
		check_callback();
		problem.post_step(polysolve::nonlinear::PostStepData(1, json::object(), x, x0));
		check_callback();

		const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(x.size(), -0.4, 0.7);
		const double eps = 1e-6;
		Eigen::VectorXd gradient, plus_gradient, minus_gradient;
		problem.gradient(x, gradient);
		CHECK(std::abs(gradient.dot(direction)
					   - (problem.value(x + eps * direction) - problem.value(x - eps * direction)) / (2 * eps))
			  < 1e-8);
		StiffnessMatrix hessian;
		problem.hessian(x, hessian);
		problem.gradient(x + eps * direction, plus_gradient);
		problem.gradient(x - eps * direction, minus_gradient);
		CHECK((hessian * direction - (plus_gradient - minus_gradient) / (2 * eps)).norm() < 1e-8);
	}
}

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
