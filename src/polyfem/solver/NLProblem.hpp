#pragma once

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/State.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <polyfem/utils/MatrixUtils.hpp>

#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/friction/friction_constraint.hpp>

#include <cppoptlib/problem.h>

namespace polyfem
{
	namespace solver
	{
		class NLProblem : public cppoptlib::Problem<double>
		{
		public:
			using typename cppoptlib::Problem<double>::Scalar;
			using typename cppoptlib::Problem<double>::TVector;
			typedef StiffnessMatrix THessian;

			NLProblem(const State &state, const assembler::RhsAssembler &rhs_assembler, const double t, const double dhat, const bool no_reduced = false);
			void init(const TVector &displacement);
			void init_time_integrator(const TVector &x_prev, const TVector &v_prev, const TVector &a_prev, const double dt);
			TVector initial_guess();

			virtual double value(const TVector &x) override;
			virtual void gradient(const TVector &x, TVector &gradv) override;
			virtual void gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv, const bool only_elastic = false);

			virtual double value(const TVector &x, const bool only_elastic);
			void gradient(const TVector &x, TVector &gradv, const bool only_elastic);

			bool is_step_valid(const TVector &x0, const TVector &x1);
			bool is_step_collision_free(const TVector &x0, const TVector &x1);
			double max_step_size(const TVector &x0, const TVector &x1);
			bool is_intersection_free(const TVector &x);

			void line_search_begin(const TVector &x0, const TVector &x1);
			void line_search_end();
			void post_step(const int iter_num, const TVector &x);

#include <polyfem/utils/DisableWarnings.hpp>
			virtual void hessian(const TVector &x, THessian &hessian);
			virtual void hessian_full(const TVector &x, THessian &gradv);
#include <polyfem/utils/EnableWarnings.hpp>

			template <class FullMat, class ReducedMat>
			static void full_to_reduced_aux(const State &state, const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced)
			{
				using namespace polyfem;

				// Reduced is already at the full size
				if (full_size == reduced_size || full.size() == reduced_size)
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
			static void reduced_to_full_aux(const State &state, const int full_size, const int reduced_size, const ReducedMat &reduced, const Eigen::MatrixXd &rhs, FullMat &full)
			{
				using namespace polyfem;

				// Full is already at the reduced size
				if (full_size == reduced_size || full_size == reduced.size())
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

			// Templated to allow VectorX* or MatrixX* input, but the size of full
			// will always be (fullsize, 1)
			template <class FullVector>
			void full_to_reduced(const FullVector &full, TVector &reduced) const
			{
				full_to_reduced_aux(state, full_size, reduced_size, full, reduced);
			}
			template <class FullVector>
			void reduced_to_full(const TVector &reduced, FullVector &full)
			{
				reduced_to_full_aux(state, full_size, reduced_size, reduced, current_rhs(), full);
			}

			void full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const;

			virtual void update_quantities(const double t, const TVector &x);
			void substepping(const double t);
			void solution_changed(const TVector &newX);

			void init_lagging(const TVector &x);
			void update_lagging(const TVector &x);
			double compute_lagging_error(const TVector &x);
			bool lagging_converged(const TVector &x);

			const Eigen::MatrixXd &current_rhs();

			virtual bool stop(const TVector &x) { return false; }

			void save_raw(const std::string &x_path, const std::string &v_path, const std::string &a_path) const;

			double heuristic_max_step(const TVector &dx);

			inline void set_ccd_max_iterations(int v) { _ccd_max_iterations = v; }

			void set_project_to_psd(bool val) { project_to_psd = val; }
			bool is_project_to_psd() const { return project_to_psd; }

			double &lagged_damping_weight() { return _lagged_damping_weight; }

			void compute_displaced_points(const TVector &full, Eigen::MatrixXd &displaced);
			void reduced_to_full_displaced_points(const TVector &reduced, Eigen::MatrixXd &displaced);

			double barrier_stiffness() const { return _barrier_stiffness; }
			const Eigen::MatrixXd &displaced_prev() const { return _displaced_prev; }
			const std::shared_ptr<const time_integrator::ImplicitTimeIntegrator> time_integrator() const { return _time_integrator; }

		protected:
			const State &state;
			bool use_adaptive_barrier_stiffness;
			double _barrier_stiffness;
			const assembler::RhsAssembler &rhs_assembler;
			bool is_time_dependent;

		private:
			const assembler::AssemblerUtils &assembler;
			Eigen::MatrixXd _current_rhs;
			StiffnessMatrix cached_stiffness;
			utils::SpareMatrixCache mat_cache;

			bool ignore_inertia;

			const int full_size, reduced_size;
			double t;
			bool rhs_computed;
			bool project_to_psd;

			double _dhat;
			double _prev_distance;
			double max_barrier_stiffness_;

			// friction variables
			double _epsv;                    ///< @brief The boundary between static and dynamic friction.
			double _mu;                      ///< @brief Coefficient of friction.
			Eigen::MatrixXd _displaced_prev; ///< @brief Displaced vertices at the start of the time-step.
			double _lagged_damping_weight;   ///< @brief Weight for lagged damping (static solve).
			TVector x_lagged;                ///< @brief The full variables from the previous lagging solve.

			ipc::BroadPhaseMethod _broad_phase_method;
			double _ccd_tolerance;
			int _ccd_max_iterations;

			double dt() const
			{
				if (_time_integrator)
				{
					assert(time_integrator()->dt() > 0);
					return time_integrator()->dt();
				}
				else
					return 1;
			}

			ipc::Constraints _constraint_set;
			ipc::FrictionConstraints _friction_constraint_set;
			ipc::Candidates _candidates;
			bool _use_cached_candidates = false;

			std::shared_ptr<time_integrator::ImplicitTimeIntegrator> _time_integrator;

			void compute_cached_stiffness();
			void update_barrier_stiffness(const TVector &full);
			void update_constraint_set(const Eigen::MatrixXd &displaced_surface);
		};
	} // namespace solver
} // namespace polyfem
