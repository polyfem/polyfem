#include "NLProblem.hpp"
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
		Eigen::VectorXd guess(reduced_size);
		guess.setZero();

		return guess;
	}

	double NLProblem::value(const TVector &x) {
		Eigen::MatrixXd full;
		reduced_to_full(x , full);

		const double elastic_energy = assembler.assemble_tensor_energy(rhs_assembler.formulation(), State::state().mesh->is_volume(), State::state().bases, State::state().bases, full);
		const double body_energy 	= rhs_assembler.compute_energy(full);

		return elastic_energy + body_energy;
	}

	void NLProblem::gradient(const TVector &x, TVector &gradv) {
		Eigen::MatrixXd full;
		reduced_to_full(x , full);

		Eigen::MatrixXd grad;
		assembler.assemble_tensor_energy_gradient(rhs_assembler.formulation(), State::state().mesh->is_volume(), State::state().n_bases, State::state().bases, State::state().bases, full, grad);
		grad -= State::state().rhs;

		full_to_reduced(grad, gradv);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian) {
		Eigen::MatrixXd full;
		reduced_to_full(x , full);

		assembler.assemble_tensor_energy_hessian(rhs_assembler.formulation(), State::state().mesh->is_volume(), State::state().n_bases, State::state().bases, State::state().bases, full, hessian);
	}

	void NLProblem::full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced)
	{
		full_to_reduced_aux(full_size, reduced_size, full, reduced);
	}

	void NLProblem::reduced_to_full(const TVector &reduced, Eigen::MatrixXd &full)
	{
		reduced_to_full_aux(full_size, reduced_size, reduced, full);
	}
}