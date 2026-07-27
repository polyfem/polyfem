#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/NavierStokesForm.hpp>
#include <polyfem/solver/forms/StackedForm.hpp>
#include <polyfem/time_integrator/BDF.hpp>

#include <polysolve/linear/Solver.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

using namespace polyfem;

namespace
{
	class TestResidualForm : public solver::Form
	{
	public:
		explicit TestResidualForm(const Eigen::VectorXd &target)
			: target_(target) {}

		std::string name() const override { return "test-residual"; }
		bool used_projected_operator() const { return used_projected_operator_; }

	protected:
		double value_unweighted(const Eigen::VectorXd &) const override
		{
			FAIL("Residual form value() must not be evaluated");
			return 0;
		}

		void first_derivative_unweighted(
			const Eigen::VectorXd &x, Eigen::VectorXd &residual) const override
		{
			residual = x - target_;
		}

		void second_derivative_unweighted(
			const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const override
		{
			used_projected_operator_ = project_to_psd_;
			jacobian.resize(x.size(), x.size());
			jacobian.setIdentity();
		}

	private:
		Eigen::VectorXd target_;
		mutable bool used_projected_operator_ = false;
	};

	StiffnessMatrix identity_matrix(const int size)
	{
		StiffnessMatrix matrix(size, size);
		matrix.setIdentity();
		return matrix;
	}
} // namespace

TEST_CASE("average pressure residual and Jacobian", "[form][navier_stokes]")
{
	solver::AveragePressureForm form(2);
	Eigen::VectorXd x(3);
	x << 2, 4, 6;

	Eigen::VectorXd residual;
	form.first_derivative(x, residual);
	CHECK(residual.isApprox((Eigen::Vector3d() << 3, 3, 3).finished()));

	StiffnessMatrix jacobian;
	form.second_derivative(x, jacobian);
	CHECK((jacobian * x).isApprox(residual));
}

TEST_CASE("stacked form propagates projected operator", "[form][navier_stokes]")
{
	auto child = std::make_shared<TestResidualForm>(Eigen::Vector2d::Zero());
	solver::StackedForm stacked;
	const auto block = stacked.add_block(2);
	stacked.add(block, child);

	stacked.set_project_to_psd(true);
	StiffnessMatrix jacobian;
	stacked.second_derivative(Eigen::Vector2d::Ones(), jacobian);
	CHECK(child->used_projected_operator());
}

TEST_CASE("residual problem value is squared residual norm", "[form][navier_stokes]")
{
	const Eigen::Vector2d target(1, -2);
	auto form = std::make_shared<TestResidualForm>(target);
	std::vector<std::shared_ptr<solver::Form>> forms = {form};
	std::vector<std::shared_ptr<solver::AugmentedLagrangianForm>> constraints;
	std::shared_ptr<polysolve::linear::Solver> linear_solver = polysolve::linear::Solver::create(
		json({{"solver", "Eigen::SparseLU"}}), logger());

	solver::NLProblem problem(
		2, nullptr, 0, forms, constraints, linear_solver,
		1, 1, identity_matrix(2), 2, /*is_residual=*/true);
	const Eigen::Vector2d x(4, 2);
	CHECK(problem.value(x) == Catch::Approx((x - target).squaredNorm()));
}

TEST_CASE("first-order inertia updates its lifted predictor", "[form][navier_stokes]")
{
	time_integrator::BDF integrator(
		1, time_integrator::ImplicitTimeIntegrator::DynamicOrder::First);
	const Eigen::Vector2d previous(1, 2);
	integrator.init(previous, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), 0.25);

	const StiffnessMatrix mass = identity_matrix(2);
	solver::InertiaForm inertia(mass, integrator);
	inertia.set_x_tilde_updater([](
									const double t, const Eigen::VectorXd &, Eigen::VectorXd &target) {
		target(0) = t;
	});
	inertia.update_quantities(3, Eigen::Vector2d::Zero());

	Eigen::VectorXd residual;
	inertia.first_derivative(Eigen::Vector2d(5, 7), residual);
	CHECK(residual.isApprox(Eigen::Vector2d(2, 5)));
	CHECK(integrator.acceleration_scaling() == Catch::Approx(0.25));
}
