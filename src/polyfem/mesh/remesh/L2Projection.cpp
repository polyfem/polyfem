#include "L2Projection.hpp"

#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/forms/ALForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/PardisoSupport>

namespace polyfem::mesh
{
	using namespace polyfem::utils;

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const State &state,
		const RhsAssembler &rhs_assembler,
		const bool is_volume,
		const int size,
		const int n_from_basis,
		const std::vector<ElementBases> &from_bases,
		const std::vector<ElementBases> &from_gbases,
		const int n_to_basis,
		const std::vector<ElementBases> &to_bases,
		const std::vector<ElementBases> &to_gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x,
		const double t0,
		const double dt,
		const int t,
		const bool lump_mass_matrix)
	{
		// solve M x = A y for x where M is the mass matrix and A is the cross mass matrix.
		Eigen::SparseMatrix<double> M, A;
		{
			MassMatrixAssembler assembler;
			Density no_density; // Density of one (i.e., no scaling of mass matrix)
			assembler.assemble(
				is_volume, size,
				n_to_basis, no_density, to_bases, to_gbases,
				cache, M);

			assembler.assemble_cross(
				is_volume, size,
				n_from_basis, from_bases, from_gbases,
				n_to_basis, to_bases, to_gbases,
				cache, A);
		}

		if (lump_mass_matrix)
		{
			M = lump_matrix(M);
		}

		// --------------------------------------------------------------------

		std::shared_ptr<L2ProjectionForm> l2_projection_form = std::make_shared<L2ProjectionForm>(M, A, y.col(0));
		std::shared_ptr<ALForm> al_form = std::make_shared<ALForm>(state, rhs_assembler, t0 + t * dt);
		std::shared_ptr<ElasticForm> elastic_form = std::make_shared<ElasticForm>(state);
		const bool use_adaptive_barrier_stiffness = !state.args["solver"]["contact"]["barrier_stiffness"].is_number();
		std::shared_ptr<ContactForm> contact_form = std::make_shared<ContactForm>(
			state,
			state.args["contact"]["dhat"],
			use_adaptive_barrier_stiffness,
			/*is_time_dependent=*/true,
			state.args["solver"]["contact"]["CCD"]["broad_phase"],
			state.args["solver"]["contact"]["CCD"]["tolerance"],
			state.args["solver"]["contact"]["CCD"]["max_iterations"]);

		if (use_adaptive_barrier_stiffness)
		{
			contact_form->set_weight(1);
			logger().debug("Using adaptive barrier stiffness");
		}
		else
		{
			contact_form->set_weight(state.args["solver"]["contact"]["barrier_stiffness"]);
			logger().debug("Using fixed barrier stiffness of {}", contact_form->barrier_stiffness());
		}

		std::vector<std::shared_ptr<Form>> forms = {l2_projection_form, al_form, elastic_form, contact_form};
		NLProblem problem(state, rhs_assembler, t0 + t * dt, forms);

		// --------------------------------------------------------------------

		// Create Newton solver
		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver;
		{
			json newton_args = state.args["solver"]["nonlinear"];
			newton_args["f_delta"] = 1e-7;
			newton_args["grad_norm"] = 1e-7;
			newton_args["use_grad_norm"] = true;
			// newton_args["relative_gradient"] = true;
			nl_solver = std::make_shared<cppoptlib::SparseNewtonDescentSolver<NLProblem>>(
				newton_args, state.args["solver"]["linear"]);
		}

		// --------------------------------------------------------------------

		// Create a lambda function to update the barrier stiffness
		auto updated_barrier_stiffness = [&](const Eigen::MatrixXd &x) {
			if (!contact_form->use_adaptive_barrier_stiffness())
				return;

			Eigen::VectorXd grad_energy(x.size(), 1);
			grad_energy.setZero();
			elastic_form->first_derivative(x, grad_energy);

			Eigen::VectorXd grad_L2(x.size());
			l2_projection_form->first_derivative(x, grad_L2);
			grad_energy += grad_L2;

			contact_form->update_barrier_stiffness(x, grad_energy);
		};

		// --------------------------------------------------------------------

		// Create augmented Lagrangian solver
		ALSolver al_solver(
			nl_solver, al_form,
			state.args["solver"]["augmented_lagrangian"]["initial_weight"],
			state.args["solver"]["augmented_lagrangian"]["max_weight"],
			updated_barrier_stiffness);

		Eigen::MatrixXd sol = Eigen::VectorXd::Zero(M.rows());
		al_solver.solve(problem, sol, state.args["solver"]["augmented_lagrangian"]["force"]);

		// --------------------------------------------------------------------

		// Construct a linear solver for M
		Eigen::PardisoLU<decltype(M)> linear_solver;
		// linear_solver->setParameters(solver_params);
		// NOTE: remove & if you want to have a more complicated LHS
		const Eigen::SparseMatrix<double> &LHS = M;
		linear_solver.analyzePattern(LHS);
		linear_solver.factorize(LHS);

		const Eigen::MatrixXd rhs = A * y.rightCols(2);
		x.resize(rhs.rows(), y.cols());
		x.col(0) = sol;
		x.rightCols(2) = linear_solver.solve(rhs);
		// x = linear_solver.solve(rhs);
		double residual_error = (LHS * x.rightCols(2) - rhs).norm();
		// double residual_error = (LHS * x - rhs).norm();
		logger().critical("residual error in L2 projection: {}", residual_error);
		assert(residual_error < 1e-12);
	}

	L2ProjectionForm::L2ProjectionForm(
		const StiffnessMatrix &M,
		const StiffnessMatrix &A,
		const Eigen::VectorXd &x_prev)
		: M_(M), rhs_(A * x_prev)
	{
	}

	double L2ProjectionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return x.transpose() * (0.5 * M_ * x - rhs_);
	}

	void L2ProjectionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = M_ * x - rhs_;
	}

	void L2ProjectionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		hessian = M_;
	}

} // namespace polyfem::mesh