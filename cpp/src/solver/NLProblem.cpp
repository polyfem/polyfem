#include "NLProblem.hpp"

#include "LinearSolver.hpp"
#include "FEMSolver.hpp"

#include "Types.hpp"

namespace poly_fem
{
	NLProblem::NLProblem(const RhsAssembler &rhs_assembler, const int full_size, const int reduced_size)
	: assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
	full_size(full_size), reduced_size(reduced_size)
	{ }


	NLProblem::NLProblem(const RhsAssembler &rhs_assembler)
	: assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
	full_size(State::state().n_bases*State::state().mesh->dimension()),
	reduced_size(State::state().n_bases*State::state().mesh->dimension() - State::state().boundary_nodes.size())
	{ }

	NLProblem::TVector NLProblem::initial_guess()
	{
		auto &state = State::state();

		auto solver = LinearSolver::create(state.args["solver_type"], state.args["precond_type"]);
		solver->setParameters(state.solver_params());
		Eigen::VectorXd b, x, guess;

		Eigen::SparseMatrix<double> A;
		assembler.assemble_tensor_problem("HookeLinearElasticity", state.mesh->is_volume(), state.n_bases, state.bases, state.iso_parametric() ? state.bases : state.geom_bases, A);

		b = state.rhs;
		state.spectrum = dirichlet_solve(*solver, A, b, state.boundary_nodes, x, true, true);

		full_to_reduced(x, guess);

		return guess;
	}

	double NLProblem::value(const TVector &x) {
		Eigen::MatrixXd full;
		reduced_to_full(x, false, full);

		const double elastic_energy = assembler.assemble_tensor_energy(rhs_assembler.formulation(), State::state().mesh->is_volume(), State::state().bases, State::state().bases, full);
		const double body_energy 	= rhs_assembler.compute_energy(full, State::state().local_neumann_boundary);

		return elastic_energy + body_energy;
	}

	void NLProblem::gradient(const TVector &x, TVector &gradv) {
		Eigen::MatrixXd full;
		reduced_to_full(x, false, full);

		Eigen::MatrixXd grad;
		assembler.assemble_tensor_energy_gradient(rhs_assembler.formulation(), State::state().mesh->is_volume(), State::state().n_bases, State::state().bases, State::state().bases, full, grad);
		grad -= State::state().rhs;

		full_to_reduced(grad, gradv);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian) {
		Eigen::MatrixXd full;
		reduced_to_full(x, false, full);

		Eigen::SparseMatrix<double> tmp;
		assembler.assemble_tensor_energy_hessian(rhs_assembler.formulation(), State::state().mesh->is_volume(), State::state().n_bases, State::state().bases, State::state().bases, full, tmp);

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

	void NLProblem::reduced_to_full(const TVector &reduced, const bool set_zero, Eigen::MatrixXd &full) const
	{
		reduced_to_full_aux(full_size, reduced_size, reduced, set_zero, full);
	}
}