#pragma once


#include "Common.hpp"
#include "LinearSolver.hpp"

#include "NLProblem.hpp"

#include "State.hpp"

#include <igl/Timer.h>
#include <Eigen/Sparse>

#include <cppoptlib/problem.h>
#include <cppoptlib/solver/lbfgssolver.h>
#include <cppoptlib/solver/isolver.h>
#include <cppoptlib/linesearch/armijo.h>

#include <cmath>



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
		{
			auto criteria = this->criteria();
			criteria.gradNorm = 1e-8;
			this->setStopCriteria(criteria);
		}

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

			grad_time = 0;
			assembly_time = 0;
			inverting_time = 0;
			linesearch_time = 0;
			igl::Timer time;

			THessian hessian;
			this->m_current.reset();
			AssemblerUtils::instance().clear_cache();

			bool analyze_pattern = true;
			do
			{
				time.start();
				objFunc.gradient(x0, grad);
				NLProblem::reduced_to_full_aux(full_size, reduced_size, grad, true, full_grad);
				time.stop();
				if(verbose)
					std::cout<<"grad time "<<time.getElapsedTimeInSec()<<std::endl;
				grad_time += time.getElapsedTimeInSec();



				time.start();
				objFunc.hessian(x0, hessian);
				hessian += (1e-5) * id;
				time.stop();
				if(verbose)
					std::cout<<"assembly time "<<time.getElapsedTimeInSec()<<std::endl;
				assembly_time += time.getElapsedTimeInSec();



        		time.start();
				poly_fem::dirichlet_solve(*solver, hessian, full_grad, State::state().boundary_nodes, full_delta_x, analyze_pattern, false);
				NLProblem::full_to_reduced_aux(full_size, reduced_size, full_delta_x, delta_x);
				delta_x *= -1;
				analyze_pattern = true;

				json tmp;
				solver->getInfo(tmp);
				internal_solver.push_back(tmp);

				if(verbose)
					std::cout<<"inverting time "<<time.getElapsedTimeInSec()<<std::endl;
				inverting_time += time.getElapsedTimeInSec();


				time.start();
				const double rate = Armijo<ProblemType, 1>::linesearch(x0, delta_x, objFunc);
				x0 += rate * delta_x;
				if(verbose)
					std::cout<<"linesearch time "<<time.getElapsedTimeInSec()<<std::endl;
				linesearch_time += time.getElapsedTimeInSec();



				++this->m_current.iterations;

				this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
				this->m_status = checkConvergence(this->m_stop, this->m_current);

				if(std::isnan(objFunc.value(x0)))
				{
					this->m_status = Status::UserDefined;
					std::cerr<<"stopping because obj func is nan"<<std::endl;
				}

				if((rate * delta_x).norm() < 1e-10)
				{
					this->m_status = Status::UserDefined;
					std::cerr<<"stopping because ||step|| is too small"<<std::endl;
				}

				if(verbose)
					std::cout << "iter: "<<this->m_current.iterations <<", rate = "<< rate<< ", f = " <<  objFunc.value(x0) << ", ||g||_inf "<< this->m_current.gradNorm <<", ||step|| "<< (rate * delta_x).norm() << std::endl;
			}
			while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));


			solver_info["internal_solver"] = internal_solver;
			solver_info["internal_solver_first"] = internal_solver.front();
			solver_info["status"] = this->status();

			const auto &crit = this->criteria();
			solver_info["iterations"] = crit.iterations;
			solver_info["xDelta"] = crit.xDelta;
			solver_info["fDelta"] = crit.fDelta;
			solver_info["gradNorm"] = crit.gradNorm;
			solver_info["condition"] = crit.condition;

			grad_time /= crit.iterations;
			assembly_time /= crit.iterations;
			inverting_time /= crit.iterations;
			linesearch_time /= crit.iterations;

			solver_info["time_grad"] = grad_time;
			solver_info["time_assembly"] = assembly_time;
			solver_info["time_inverting"] = inverting_time;
			solver_info["time_linesearch"] = linesearch_time;
		}

		void getInfo(json &params)
		{
			params = solver_info;
		}

	private:
		const bool verbose;
		json solver_info;
		json internal_solver = json::array();

		double grad_time;
		double assembly_time;
		double inverting_time;
		double linesearch_time;

	};
}
