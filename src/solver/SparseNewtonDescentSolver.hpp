#pragma once

#include <polyfem/Common.hpp>
#include <polysolve/LinearSolver.hpp>
#include <polyfem/NLProblem.hpp>
#include <polyfem/MatrixUtils.hpp>
#include <polyfem/State.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#include <cppoptlib/problem.h>
#include <cppoptlib/solver/isolver.h>
#include <cppoptlib/linesearch/armijo.h>
#include <cppoptlib/linesearch/morethuente.h>

#include <cmath>

namespace cppoptlib
{

	template <typename ProblemType>
	class SparseNewtonDescentSolver : public ISolver<ProblemType, 2>
	{
	public:
		using Superclass = ISolver<ProblemType, 2>;

		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		enum class LineSearch
		{
			Armijo,
			ArmijoAlt,
			Bisection,
			MoreThuente,
			None
		};

		SparseNewtonDescentSolver(const json &solver_param, const std::string &solver_type, const std::string &precond_type)
			: solver_param(solver_param), solver_type(solver_type), precond_type(precond_type)
		{
			auto criteria = this->criteria();
			criteria.fDelta = solver_param.count("fDelta") ? double(solver_param["fDelta"]) : 1e-9;
			criteria.gradNorm = solver_param.count("gradNorm") ? double(solver_param["gradNorm"]) : 1e-8;
			criteria.iterations = solver_param.count("nl_iterations") ? int(solver_param["nl_iterations"]) : 100;

			use_gradient_norm_ = solver_param.count("useGradNorm") ? bool(solver_param["useGradNorm"]) : true;
			this->setStopCriteria(criteria);
		}

		void setLineSearch(const std::string &name)
		{
			if (name == "armijo")
			{
				line_search = LineSearch::Armijo;
			}
			else if (name == "armijo_alt")
			{
				line_search = LineSearch::ArmijoAlt;
			}
			else if (name == "bisection")
			{
				line_search = LineSearch::Bisection;
			}
			else if (name == "more_thuente")
			{
				line_search = LineSearch::MoreThuente;
			}
			else if (name == "none")
			{
				line_search = LineSearch::None;
			}
			else
			{
				polyfem::logger().error("[SparseNewtonDescentSolver] Unknown line search.");
				throw std::invalid_argument("[SparseNewtonDescentSolver] Unknown line search.");
			}

			polyfem::logger().debug("\tline search {}", name);
			solver_info["line_search"] = name;
		}

		double armijo_linesearch(const TVector &x, const TVector &searchDir, ProblemType &objFunc, const double alpha_init = 1.0)
		{
			static const int MAX_STEP_SIZE_ITER = 12;

			const double c = 0.5;
			const double tau = 0.5;

			TVector x1 = x + alpha_init * searchDir;
			double alpha = std::min(alpha_init, objFunc.max_step_size(x, x1));
			// polyfem::logger().trace("inital step {}", step_size);
			if (alpha != alpha_init)
			{
				x1 = x + alpha * searchDir;
			}
			double f = objFunc.value(x1);
			const double f_in = objFunc.value(x);

			TVector grad(x.rows());
			objFunc.gradient(x, grad);

			const double Cache = c * grad.dot(searchDir);

			int cur_iter = 0;
#ifndef NDEBUG
			//max_step_size should return a collision free step
			bool valid = objFunc.is_step_valid(x, x1);
			assert(valid);
#else
			bool valid = true;
#endif
			while ((std::isinf(f) || std::isnan(f) || f > f_in + alpha * Cache || !valid) && alpha > 1e-7 && cur_iter <= MAX_STEP_SIZE_ITER)
			{
				alpha *= tau;
				x1 = x + alpha * searchDir;
				f = objFunc.value(x1);

				cur_iter++;
#ifndef NDEBUG
				//max_step_size should return a collision free step
				valid = objFunc.is_step_valid(x, x1);
				assert(valid);
#endif
			}

			// std::cout << cur_iter << " " << MAX_STEP_SIZE_ITER << " " << alpha << std::endl;

			if (alpha <= 1e-7)
				return std::nan("");
			else
				return alpha;
		}

		double linesearch(const TVector &x, const TVector &grad, ProblemType &objFunc)
		{
			static const int MAX_STEP_SIZE_ITER = 25;

			const double old_energy = objFunc.value(x);
			int cur_iter = 0;

			TVector new_x = x + grad;
			double step_size = std::min(1., objFunc.max_step_size(x, new_x));
			polyfem::logger().trace("inital step {}", step_size);
			if (step_size != 1)
			{
				new_x = x + step_size * grad;
			}

			// std::cout<<"grad\n"<<grad<<std::endl;

			while (step_size > 1e-7 || cur_iter < MAX_STEP_SIZE_ITER)
			{
				double cur_e = objFunc.value(new_x);
#ifndef NDEBUG
				//max_step_size should return a collision free step
				const bool valid = objFunc.is_step_valid(x, new_x);
				assert(valid);
#else
				const bool valid = true;
#endif
				polyfem::logger().trace("ls it: {} delta: {} invalid: {} ", cur_iter, (cur_e - old_energy), !valid);
				if (std::isinf(cur_e) || std::isnan(cur_e) || cur_e >= old_energy || !valid)
				// if (std::isinf(cur_e) || std::isnan(cur_e) || (cur_e >= old_energy && fabs(cur_e - old_energy) > 1e-7) || !valid)
				{
					step_size /= 2.;
					new_x = x + step_size * grad;
				}
				else
				{
					return step_size;
				}
				cur_iter++;
			}

			// return step_size;
			return std::nan("");
		}

		void minimize(ProblemType &objFunc, TVector &x0)
		{
			using namespace polyfem;
			// const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();
			// const int precond_num = problem_dim * state.n_bases;

			// const json &params = State::state().solver_params();
			// auto solver = LinearSolver::create(State::state().solver_type(), State::state().precond_type());

			auto solver = polysolve::LinearSolver::create(solver_type, precond_type);
			solver->setParameters(solver_param);
			polyfem::logger().debug("\tinternal solver {}", solver->name());

			const int reduced_size = x0.rows();

			polyfem::StiffnessMatrix id(reduced_size, reduced_size);
			id.setIdentity();

			TVector grad = TVector::Zero(reduced_size);
			// TVector full_grad;

			// TVector full_delta_x;
			TVector delta_x(reduced_size);
			delta_x.setZero();

			grad_time = 0;
			assembly_time = 0;
			inverting_time = 0;
			linesearch_time = 0;
			igl::Timer time;

			polyfem::StiffnessMatrix hessian;
			this->m_current.reset();
			AssemblerUtils::instance().clear_cache();

			size_t next_hessian = 0;
			// double factor = 1e-5;
			double old_energy = std::nan("");
			double first_energy = std::nan("");
			error_code_ = 0;

			time.start();
			objFunc.gradient(x0, grad);
			time.stop();

			polyfem::logger().debug("\tgrad time {}s norm: {}", time.getElapsedTimeInSec(), grad.norm());
			grad_time += time.getElapsedTimeInSec();
			bool line_search_failed = false;

			// std::cout<<"x0\n"<<x0<<std::endl;

			if (std::isnan(grad.norm()))
			{
				this->m_status = Status::UserDefined;
				polyfem::logger().debug("stopping because first grad is nan");
				error_code_ = -10;
				return;
			}

			do
			{
				const size_t iter = this->m_current.iterations;
				bool new_hessian = iter == next_hessian;

				if (new_hessian && !line_search_failed)
				{
					time.start();
					objFunc.hessian(x0, hessian);
					// hessian = 1e-8 * id;
					//factor *= 1e-1;
					time.stop();
					polyfem::logger().debug("\tassembly time {}s", time.getElapsedTimeInSec());
					assembly_time += time.getElapsedTimeInSec();

					next_hessian += 1;

					if (iter == 0)
					{
						if (has_hessian_nans(hessian))
						{
							this->m_status = Status::UserDefined;
							polyfem::logger().debug("stopping because hessian is nan");
							error_code_ = -10;
							break;
						}
					}
				}

				// std::cout<<hessian<<std::endl;
				time.start();

				if (new_hessian && !line_search_failed)
				{
					//TODO: get the correct size
					solver->analyzePattern(hessian, hessian.rows());
					solver->factorize(hessian);
				}
				if (!line_search_failed)
					solver->solve(grad, delta_x);

				//gradient descent, check descent direction
				const double residual = (hessian * delta_x - grad).norm();
				if (line_search_failed || std::isnan(residual) || residual > 1e-7)
				{
					polyfem::logger().debug("\treverting to gradient descent, since residual is {}", residual);
					delta_x = grad;
				}

				delta_x *= -1;

				json tmp;
				solver->getInfo(tmp);
				internal_solver.push_back(tmp);

				polyfem::logger().debug("\tinverting time {}s", time.getElapsedTimeInSec());
				inverting_time += time.getElapsedTimeInSec();

				time.start();

				double rate;
				switch (line_search)
				{
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
				case LineSearch::None:
					rate = 1e-1;
					break;
				}

				if (std::isnan(rate))
				{
					if (!line_search_failed)
					{
						line_search_failed = true;
						polyfem::logger().debug("\tline search failed, reverting to gradient descent");
						this->m_status = Status::Continue;
						continue;
					}
				}
				line_search_failed = false;

				x0 += rate * delta_x;
				time.start();
				objFunc.gradient(x0, grad);
				time.stop();

				polyfem::logger().debug("\tgrad time {}s norm: {}", time.getElapsedTimeInSec(), grad.norm());
				grad_time += time.getElapsedTimeInSec();

				polyfem::logger().debug("\tlinesearch time {}s", time.getElapsedTimeInSec());
				linesearch_time += time.getElapsedTimeInSec();

				++this->m_current.iterations;

				const double energy = objFunc.value(x0);
				const double step = (rate * delta_x).norm();

				this->m_current.fDelta = 1; //std::abs(old_energy - energy) / std::abs(old_energy);
				this->m_current.gradNorm = use_gradient_norm_ ? grad.norm() : delta_x.norm();
				this->m_status = checkConvergence(this->m_stop, this->m_current);
				old_energy = energy;
				if (std::isnan(first_energy))
				{
					first_energy = energy;
				}

				if (std::isnan(energy) || std::isinf(energy))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().debug("stopping because obj func is nan or inf");
					error_code_ = -10;
				}

				if (this->m_status == Status::Continue && step < 1e-10)
				{
					if (new_hessian)
					{
						this->m_status = Status::UserDefined;
						polyfem::logger().debug("stopping because ||step||={} is too small", step);
						error_code_ = -1;
					}
					else
					{
						next_hessian = this->m_current.iterations;
						polyfem::logger().debug("\tstep small force recompute hessian");
					}
				}

				//if(rate >= 1 && next_hessian == this->m_current.iterations)
				//	next_hessian += 2;

				polyfem::logger().debug("\titer: {}, f = {}, ||g||_2 = {}, rate = {}, ||step|| = {}, dot = {}",
										this->m_current.iterations, energy, this->m_current.gradNorm, rate, step, delta_x.dot(grad) / grad.norm());
				delta_x *= -1;
			} while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));

			polyfem::logger().info("Newton finished niters = {}, f = {}, ||g||_2 = {}", this->m_current.iterations, old_energy, this->m_current.gradNorm);

			if (error_code_ != -10)
			{
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
			}

			solver_info["time_grad"] = grad_time;
			solver_info["time_assembly"] = assembly_time;
			solver_info["time_inverting"] = inverting_time;
			solver_info["time_linesearch"] = linesearch_time;
		}

		void getInfo(json &params)
		{
			params = solver_info;
		}

		int error_code() const { return error_code_; }

	private:
		const json solver_param;
		const std::string solver_type;
		const std::string precond_type;

		int error_code_;
		bool use_gradient_norm_;
		json solver_info;

		json internal_solver = json::array();

		LineSearch line_search = LineSearch::Armijo;

		double grad_time;
		double assembly_time;
		double inverting_time;
		double linesearch_time;

		bool has_hessian_nans(const polyfem::StiffnessMatrix &hessian)
		{
			for (int k = 0; k < hessian.outerSize(); ++k)
			{
				for (polyfem::StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
				{
					if (std::isnan(it.value()))
						return true;
				}
			}

			return false;
		}
	};
} // namespace cppoptlib
