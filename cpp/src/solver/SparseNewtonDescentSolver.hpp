#pragma once


#include <polyfem/Common.hpp>
#include <polyfem/LinearSolver.hpp>
#include <polyfem/NLProblem.hpp>
#include <polyfem/MatrixUtils.hpp>
#include <polyfem/State.hpp>

#include <igl/Timer.h>
#include <igl/line_search.h>
#include <Eigen/Sparse>

#include <cppoptlib/problem.h>
#include <cppoptlib/solver/lbfgssolver.h>
#include <cppoptlib/solver/isolver.h>
#include <cppoptlib/linesearch/armijo.h>
#include <cppoptlib/linesearch/morethuente.h>


#include <cmath>



namespace cppoptlib {

	template<typename ProblemType>
	class SparseNewtonDescentSolver : public ISolver<ProblemType, 2> {
	public:
		using Superclass = ISolver<ProblemType, 2>;

		using typename Superclass::Scalar;
		using typename Superclass::TVector;
		typedef Eigen::SparseMatrix<double> THessian;

		enum class LineSearch {
			Armijo,
			ArmijoAlt,
			Bisection,
			MoreThuente,
		};

		SparseNewtonDescentSolver(const bool verbose)
		: verbose(verbose)
		{
			auto criteria = this->criteria();
			criteria.fDelta = 1e-9;
			criteria.gradNorm = 1e-9;
			criteria.iterations = 100;
			this->setStopCriteria(criteria);
		}

		void setLineSearch(const std::string &name) {
			if (name ==  "armijo") {
				line_search = LineSearch::Armijo;
			} else if (name == "armijo_alt") {
				line_search = LineSearch::ArmijoAlt;
			} else if (name == "bisection") {
				line_search = LineSearch::Bisection;
			} else if (name == "more_thuente") {
				line_search = LineSearch::MoreThuente;
			} else {
				throw std::invalid_argument("[SparseNewtonDescentSolver] Unknown line search.");
			}
			if(verbose)
				std::cout<<"\tline search "<<name<<std::endl;
			solver_info["line_search"] = name;
		}

		double armijo_linesearch(const TVector &x, const TVector &searchDir, ProblemType &objFunc, const double alpha_init = 1.0)
		{
			static const int MAX_STEP_SIZE_ITER = 12;

			const double c = 0.5;
			const double tau = 0.5;

			double alpha = alpha_init;
			double f = objFunc.value(x + alpha * searchDir);
			const double f_in = objFunc.value(x);

			TVector grad(x.rows());
			objFunc.gradient(x, grad);

			const double Cache = c * grad.dot(searchDir);

			int cur_iter = 0;
			while(f > f_in + alpha * Cache && cur_iter < MAX_STEP_SIZE_ITER) {
				alpha *= tau;
				f = objFunc.value(x + alpha * searchDir);

				cur_iter++;
			}

			return alpha;
		}


		double linesearch(const TVector &x, const TVector &grad, ProblemType &objFunc)
		{
			static const int MAX_STEP_SIZE_ITER = 12;

			const double old_energy = objFunc.value(x);
			double new_energy = old_energy;
			int cur_iter = 0;

			double step_size = 1;

			while (new_energy >= old_energy && cur_iter < MAX_STEP_SIZE_ITER)
			{
				const TVector new_x = x + step_size * grad;

				double cur_e = objFunc.value(new_x);
				if ( cur_e >= old_energy)
				{
					step_size /= 2.;
				}
				else
				{
					return step_size;
				}
				cur_iter++;
			}

			return step_size;
		}

		void minimize(ProblemType &objFunc, TVector &x0) {
			using namespace polyfem;

			const json &params = State::state().solver_params();
			auto solver = LinearSolver::create(State::state().solver_type(), State::state().precond_type());
			solver->setParameters(params);
			if(verbose)
				std::cout<<"\tinternal solver "<<solver->name()<<std::endl;

			const int reduced_size = x0.rows();

			THessian id(reduced_size, reduced_size);
			id.setIdentity();

			TVector grad = TVector::Zero(reduced_size);
			// TVector full_grad;

			// TVector full_delta_x;
			TVector delta_x(reduced_size);

			grad_time = 0;
			assembly_time = 0;
			inverting_time = 0;
			linesearch_time = 0;
			igl::Timer time;

			THessian hessian;
			this->m_current.reset();
			AssemblerUtils::instance().clear_cache();

			size_t next_hessian = 0;
			double factor = 1e-5;
			double old_energy = std::nan("");
			double first_energy = std::nan("");
			do
			{
				time.start();
				objFunc.gradient(x0, grad);
				time.stop();

				if(verbose)
					std::cout<<"\tgrad time "<<time.getElapsedTimeInSec()<<std::endl;
				grad_time += time.getElapsedTimeInSec();

				int iter = this->m_current.iterations;
				bool new_hessian = this->m_current.iterations == next_hessian;

				if(new_hessian)
				{
					time.start();
					objFunc.hessian(x0, hessian);
					hessian += 0.0 * id;
					//factor *= 1e-1;
					time.stop();
					if(verbose)
						std::cout<<"\tassembly time "<<time.getElapsedTimeInSec()<<std::endl;
					assembly_time += time.getElapsedTimeInSec();

					next_hessian += 5;
				}

				// std::cout<<hessian<<std::endl;
				time.start();

				if(new_hessian)
				{
					solver->analyzePattern(hessian);
					solver->factorize(hessian);
				}
				solver->solve(grad, delta_x);

				delta_x *= -1;

				json tmp;
				solver->getInfo(tmp);
				internal_solver.push_back(tmp);

				if(verbose)
					std::cout<<"\tinverting time "<<time.getElapsedTimeInSec()<<std::endl;
				inverting_time += time.getElapsedTimeInSec();


				time.start();

				double rate;
				switch (line_search) {
					case LineSearch::Armijo:
						rate = armijo_linesearch(x0, delta_x, objFunc);
						break;
					case LineSearch::ArmijoAlt:
						rate = Armijo<ProblemType, 1>::linesearch(x0, delta_x, objFunc);
						break;
					case LineSearch::Bisection:
						rate = linesearch(x0, delta_x, objFunc);
						break;
					case LineSearch::MoreThuente:
						rate = MoreThuente<ProblemType, 1>::linesearch(x0, delta_x, objFunc);
						break;
				}

				x0 += rate * delta_x;


				if(verbose)
					std::cout<<"\tlinesearch time "<<time.getElapsedTimeInSec()<<std::endl;
				linesearch_time += time.getElapsedTimeInSec();


				++this->m_current.iterations;

				const double energy = objFunc.value(x0);
				const double step = (rate * delta_x).norm();

				this->m_current.fDelta = std::abs(old_energy - energy) / std::abs(first_energy);
				this->m_current.gradNorm = grad.norm();
				this->m_status = checkConvergence(this->m_stop, this->m_current);
				old_energy = energy;
				if (std::isnan(first_energy)) { first_energy = energy; }

				if(std::isnan(energy))
				{
					this->m_status = Status::UserDefined;
					std::cerr<<"stopping because obj func is nan"<<std::endl;
				}

				if(this->m_status == Status::Continue && step < 1e-10)
				{
					if(new_hessian)
					{
						this->m_status = Status::UserDefined;
						std::cerr<<"stopping because ||step||=" << step << " is too small"<<std::endl;
					}
					else
					{
						next_hessian = this->m_current.iterations;
						if(verbose)
							std::cout<<"\tstep small force recompute hessian"<<std::endl;
					}
				}

				if(verbose) {
					tfm::printf("\titer: %s, f = %s, ‖g‖_2 = %s, rate = %s, ‖step‖ = %s, dot = %s\n",
						this->m_current.iterations, energy, this->m_current.gradNorm, rate, step, delta_x.dot(grad)/grad.norm());
					// tfm::printf("\tspectrum: %s (%s)\n", spectrum(3) / spectrum(0), spectrum.transpose());
					std::cout << this->criteria() << std::endl;
				}
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

		LineSearch line_search = LineSearch::Armijo;

		double grad_time;
		double assembly_time;
		double inverting_time;
		double linesearch_time;

	};
}
