#pragma once

#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/RhsAssembler.hpp>
#include <polyfem/State.hpp>

#include <polyfem/MatrixUtils.hpp>

#include <ipc/collision_constraint.hpp>

#include <cppoptlib/problem.h>

namespace polyfem
{
	class NLProblem : public cppoptlib::Problem<double>
	{
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;
		typedef StiffnessMatrix THessian;

		NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool project_to_psd, const bool no_reduced = false);
		void init(const TVector &displacement);
		void init_timestep(const TVector &x_prev, const TVector &v_prev, const TVector &a_prev, const double dt);
		TVector initial_guess();

		virtual double value(const TVector &x) override;
		virtual void gradient(const TVector &x, TVector &gradv) override;
		virtual void gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv, const bool only_elastic = false);

		virtual double value(const TVector &x, const bool only_elastic);
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic);

		bool is_step_valid(const TVector &x0, const TVector &x1);
		bool is_step_collision_free(const TVector &x0, const TVector &x1);
		double max_step_size(const TVector &x0, const TVector &x1);

		void line_search_begin(const TVector &x0, const TVector &x1);
		void line_search_end();
		void post_step(const TVector &x0);

#include <polyfem/DisableWarnings.hpp>
		virtual void hessian(const TVector &x, THessian &hessian);
		virtual void hessian_full(const TVector &x, THessian &gradv);
#include <polyfem/EnableWarnings.hpp>

		template <class FullMat, class ReducedMat>
		static void full_to_reduced_aux(State &state, const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced)
		{
			using namespace polyfem;

			if (full_size == reduced_size)
			{
				reduced = full;
				return;
			}

			assert(full.size() == full_size);
			assert(full.cols() == 1);
			reduced.resize(reduced_size, 1);

			long j = 0;
			size_t k = 0;
			for (int i = 0; i < full.size(); ++i)
			{
				if (k < state.boundary_nodes.size() && state.boundary_nodes[k] == i)
				{
					++k;
					continue;
				}

				reduced(j++) = full(i);
			}
			assert(j == reduced.size());
		}

		template <class ReducedMat, class FullMat>
		static void reduced_to_full_aux(State &state, const int full_size, const int reduced_size, const ReducedMat &reduced, const Eigen::MatrixXd &rhs, FullMat &full)
		{
			using namespace polyfem;

			if (full_size == reduced_size)
			{
				full = reduced;
				return;
			}

			assert(reduced.size() == reduced_size);
			assert(reduced.cols() == 1);
			full.resize(full_size, 1);

			long j = 0;
			size_t k = 0;
			for (int i = 0; i < full.size(); ++i)
			{
				if (k < state.boundary_nodes.size() && state.boundary_nodes[k] == i)
				{
					++k;
					full(i) = rhs(i);
					continue;
				}

				full(i) = reduced(j++);
			}

			assert(j == reduced.size());
		}

		void full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced) const;
		void reduced_to_full(const TVector &reduced, Eigen::MatrixXd &full);

		virtual void update_quantities(const double t, const TVector &x);
		void substepping(const double t);
		void solution_changed(const TVector &newX);

		const Eigen::MatrixXd &current_rhs();

		virtual bool stop(const TVector &x) { return false; }

	protected:
		State &state;
		double _barrier_stiffness;
		void compute_displaced_points(const Eigen::MatrixXd &full, Eigen::MatrixXd &displaced);
		const RhsAssembler &rhs_assembler;
		bool is_time_dependent;

	private:
		AssemblerUtils &assembler;
		Eigen::MatrixXd _current_rhs;
		StiffnessMatrix cached_stiffness;
		SpareMatrixCache mat_cache;

		const int full_size, reduced_size;
		double t;
		bool rhs_computed;
		bool project_to_psd;

		double _dhat;
		double _prev_distance;
		double max_barrier_stiffness_;

		double dt;
		TVector x_prev, v_prev, a_prev;

		ipc::Constraints _constraint_set;
		ipc::Candidates _candidates;

		void compute_cached_stiffness();
		void update_barrier_stiffness(const TVector &full);
	};
} // namespace polyfem
