#include <polyfem/NLProblem.hpp>

#include <polyfem/LinearSolver.hpp>
#include <polyfem/FEMSolver.hpp>

#include <polyfem/Types.hpp>

#include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
	NLProblem::NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const int full_size, const int reduced_size)
	: state(state), assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
	full_size(full_size), reduced_size(reduced_size), t(t), rhs_computed(false)
	{ }


	NLProblem::NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t)
	: state(state), assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
	full_size(state.n_bases*state.mesh->dimension()),
	reduced_size(state.n_bases*state.mesh->dimension() - state.boundary_nodes.size()),
	t(t), rhs_computed(false)
	{ }

	// NLProblem::TVector NLProblem::initial_guess()
	// {
	// 	Eigen::VectorXd guess(reduced_size);
	// 	guess.setZero();

	// 	return guess;

 // 	// 	auto solver = LinearSolver::create(state.args["solver_type"], state.args["precond_type"]);
	// 	// solver->setParameters(state.solver_params());
	// 	// Eigen::VectorXd b, x, guess;
 // 	// 	THessian A;
	// 	// assembler.assemble_problem("LinearElasticity", state.mesh->is_volume(), state.n_bases, state.bases, state.iso_parametric() ? state.bases : state.geom_bases, A);
 // 	// 	if(!rhs_computed)
	// 	// {
	// 	// 	rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, state.rhs, t, current_rhs);
	// 	// 	rhs_computed = true;
	// 	// }

	// 	// b = current_rhs;
	// 	// dirichlet_solve(*solver, A, b, state.boundary_nodes, x, "", false);
 // 	// 	full_to_reduced(x, guess);

 // 	// 	return guess;
	// }

	const Eigen::MatrixXd &NLProblem::current_rhs()
	{
		if(!rhs_computed)
		{
			rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, state.rhs, t, _current_rhs);
			rhs_computed = true;
		}

		return _current_rhs;
	}

	double NLProblem::value(const TVector &x) {
		Eigen::MatrixXd full;
		if(x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		const double elastic_energy = assembler.assemble_energy(rhs_assembler.formulation(), state.mesh->is_volume(), state.bases, gbases, full);
		const double body_energy 	= rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.args["n_boundary_samples"], t);

		return elastic_energy + body_energy;
	}

	void NLProblem::gradient(const TVector &x, TVector &gradv) {
		Eigen::MatrixXd full;
		if(x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		Eigen::MatrixXd grad;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, grad);
		grad -= current_rhs();

		full_to_reduced(grad, gradv);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian) {
		Eigen::MatrixXd full;
		if(x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;

		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		THessian tmp;
		assembler.assemble_energy_hessian(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, tmp);
		// Eigen::saveMarket(tmp, "tmp.mat");
		// exit(0);

		std::vector< Eigen::Triplet<double> > entries;

		Eigen::VectorXi indices(full.size());

		int index = 0;
		size_t kk = 0;
		for(int i = 0; i < full.size(); ++i)
		{
			if(kk < state.boundary_nodes.size() && state.boundary_nodes[kk] == i)
			{
				++kk;
				indices(i) = -1;
				continue;
			}

			indices(i) = index++;
		}
		assert(index == reduced_size);

		for (int k = 0; k < tmp.outerSize(); ++k) {
			if(indices(k) < 0)
			{
				continue;
			}

			for (THessian::InnerIterator it(tmp, k); it; ++it)
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

		hessian.resize(reduced_size, reduced_size);
		hessian.setFromTriplets(entries.begin(), entries.end());
		hessian.makeCompressed();
	}

	void NLProblem::full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced) const
	{
		full_to_reduced_aux(state, full_size, reduced_size, full, reduced);
	}

	void NLProblem::reduced_to_full(const TVector &reduced, Eigen::MatrixXd &full)
	{
		reduced_to_full_aux(state, full_size, reduced_size, reduced, current_rhs(), full);
	}
}