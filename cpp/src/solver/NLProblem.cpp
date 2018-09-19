#include <polyfem/NLProblem.hpp>

#include <polyfem/LinearSolver.hpp>
#include <polyfem/FEMSolver.hpp>

#include <polyfem/Types.hpp>

#include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
	NLProblem::NLProblem(const RhsAssembler &rhs_assembler, const double t, const int full_size, const int reduced_size)
	: assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
	full_size(full_size), reduced_size(reduced_size), t(t), rhs_computed(false)
	{ }


	NLProblem::NLProblem(const RhsAssembler &rhs_assembler, const double t)
	: assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
	full_size(State::state().n_bases*State::state().mesh->dimension()),
	reduced_size(State::state().n_bases*State::state().mesh->dimension() - State::state().boundary_nodes.size()),
	t(t), rhs_computed(false)
	{ }

	NLProblem::TVector NLProblem::initial_guess()
	{
		// Eigen::VectorXd guess(reduced_size);
		// guess.setZero();

		// return guess;

		const auto &state = State::state();
 		auto solver = LinearSolver::create(state.args["solver_type"], state.args["precond_type"]);
		solver->setParameters(state.solver_params());
		Eigen::VectorXd b, x, guess;
 		Eigen::SparseMatrix<double> A;
		assembler.assemble_problem("LinearElasticity", state.mesh->is_volume(), state.n_bases, state.bases, state.iso_parametric() ? state.bases : state.geom_bases, A);
 		b = state.rhs * t;
		dirichlet_solve(*solver, A, b, state.boundary_nodes, x, "", false);
 		full_to_reduced(x, guess);

 		return guess;
	}

	double NLProblem::value(const TVector &x) {
		Eigen::MatrixXd full;
		reduced_to_full(x, full);

		const auto &state = State::state();
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		const double elastic_energy = assembler.assemble_energy(rhs_assembler.formulation(), state.mesh->is_volume(), state.bases, gbases, full);
		const double body_energy 	= rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.args["n_boundary_samples"], t);

		return elastic_energy + body_energy;
	}

	void NLProblem::gradient(const TVector &x, TVector &gradv) {
		const auto &state = State::state();

		Eigen::MatrixXd full;
		reduced_to_full(x, full);
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		Eigen::MatrixXd grad;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, grad);
		if(!rhs_computed)
		{
			rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, state.rhs, t, current_rhs);
			rhs_computed = true;
		}
		grad -= current_rhs;

		full_to_reduced(grad, gradv);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian) {
		const auto &state = State::state();
		Eigen::MatrixXd full;
		reduced_to_full(x, full);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		Eigen::SparseMatrix<double> tmp;
		assembler.assemble_energy_hessian(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, tmp);
		// Eigen::saveMarket(tmp, "tmp.mat");
		// exit(0);

		std::vector< Eigen::Triplet<double> > entries;

		Eigen::VectorXi indices(full.size());

		int index = 0;
		size_t kk = 0;
		for(int i = 0; i < full.size(); ++i)
		{
			if(kk < State::state().boundary_nodes.size() && State::state().boundary_nodes[kk] == i)
			{
				++kk;
				indices(i) = -1;
				continue;
			}

			indices(i) = index++;
		}
		assert(index == x.size());

		for (int k = 0; k < tmp.outerSize(); ++k) {
			if(indices(k) < 0)
			{
				continue;
			}

			for (Eigen::SparseMatrix<double>::InnerIterator it(tmp, k); it; ++it)
			{
				// std::cout<<it.row()<<" "<<it.col()<<" "<<k<<std::endl;
				assert(it.col() == k);
				if(indices(it.row()) < 0 || indices(it.col()) < 0)
				{
					continue;
				}

				assert(indices(it.row()) >= 0);
				assert(indices(it.col()) >= 0);

				entries.emplace_back(indices(it.row()),indices(it.col()), it.value());
			}
		}

		hessian.resize(x.size(),x.size());
		hessian.setFromTriplets(entries.begin(), entries.end());
		hessian.makeCompressed();
	}

	void NLProblem::full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced) const
	{
		full_to_reduced_aux(full_size, reduced_size, full, reduced);
	}

	void NLProblem::reduced_to_full(const TVector &reduced, Eigen::MatrixXd &full)
	{
		if(!rhs_computed)
		{
			const auto &state = State::state();
			rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, state.rhs, t, current_rhs);
			rhs_computed = true;
		}

		reduced_to_full_aux(full_size, reduced_size, reduced, current_rhs, full);
	}
}