#pragma once


#include "Common.hpp"
#include "LinearSolver.hpp"

#include "NLProblem.hpp"

#include "State.hpp"

#include <Eigen/Sparse>

#include <cppoptlib/problem.h>
#include <cppoptlib/solver/lbfgssolver.h>
#include <cppoptlib/solver/isolver.h>
#include <cppoptlib/linesearch/armijo.h>


namespace cppoptlib {

	template<typename ProblemType>
	class SparseNewtonDescentSolver : public ISolver<ProblemType, 2> {
	public:
		using Superclass = ISolver<ProblemType, 2>;

		using typename Superclass::Scalar;
		using typename Superclass::TVector;
		typedef Eigen::SparseMatrix<double> THessian;

		SparseNewtonDescentSolver(const bool verbose)
		: verbose(verbose)
		{ }

		void minimize(ProblemType &objFunc, TVector &x0) {
			using namespace poly_fem;

			const json &params = State::state().solver_params();
			auto solver = LinearSolver::create(State::state().solver_type(), State::state().precond_type());
			solver->setParameters(params);

			const int reduced_size = x0.rows();
			const int full_size = State::state().n_bases*State::state().mesh->dimension();
			assert(full_size == reduced_size + State::state().boundary_nodes.size());

			THessian id(full_size, full_size);
			id.setIdentity();


			TVector grad = TVector::Zero(reduced_size);
			TVector full_grad;

			TVector full_delta_x;
			TVector delta_x;

			THessian hessian(reduced_size, reduced_size);
			this->m_current.reset();
			for(int iter = 0; iter < 15; ++iter)
			// do
			{
				objFunc.gradient(x0, grad);
				NLProblem::reduced_to_full_aux(full_size, reduced_size, grad, true, full_grad);

				objFunc.hessian(x0, hessian);
				hessian += (1e-5) * id;

				// std::cout<<x0<<std::endl;
				// std::cout<<grad<<std::endl;


        		// TVector delta_x = hessian.lu().solve(-grad);
				poly_fem::dirichlet_solve(*solver, hessian, full_grad, State::state().boundary_nodes, full_delta_x);
				NLProblem::full_to_reduced_aux(full_size, reduced_size, full_delta_x, delta_x);
				delta_x *= -1;


				const double rate = Armijo<ProblemType, 1>::linesearch(x0, delta_x, objFunc);
				x0 += rate * delta_x;

				++this->m_current.iterations;

				this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
				this->m_status = checkConvergence(this->m_stop, this->m_current);

				if(verbose)
					std::cout << "iter: "<<this->m_current.iterations <<", rate = "<< rate<< ", f = " <<  objFunc.value(x0) << ", ||g||_inf "<< this->m_current.gradNorm <<", ||step|| "<< (rate * delta_x).norm() << std::endl;
			}
			// while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));
		}

	private:
		const bool verbose;
	};
}