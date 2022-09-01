#pragma once

#include <polyfem/solver/forms/Form.hpp>

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

			NLProblem(const State &state, const assembler::RhsAssembler &rhs_assembler, const double t, std::vector<std::shared_ptr<Form>> &forms);
			void init(const TVector &displacement);

			virtual double value(const TVector &x) override;
			virtual void gradient(const TVector &x, TVector &gradv) override;
#include <polyfem/utils/DisableWarnings.hpp>
			virtual void hessian(const TVector &x, THessian &hessian);
#include <polyfem/utils/EnableWarnings.hpp>

			bool is_step_valid(const TVector &x0, const TVector &x1);
			double max_step_size(const TVector &x0, const TVector &x1);

			void line_search_begin(const TVector &x0, const TVector &x1);
			void line_search_end();
			void post_step(const int iter_num, const TVector &x);

			void set_project_to_psd(bool val);

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
				full_to_reduced_aux(state_, full_size, reduced_size, full, reduced);
			}

			template <class FullVector>
			void reduced_to_full(const TVector &reduced, FullVector &full)
			{
				Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(full_size, 1);

				if (reduced_size != full_size)
				{
					// rhs_assembler.set_bc(state_.local_boundary, state_.boundary_nodes, state_.n_boundary_samples(), state_.local_neumann_boundary, tmp, t_);
					rhs_assembler_.set_bc(state_.local_boundary, state_.boundary_nodes, state_.n_boundary_samples(), std::vector<mesh::LocalBoundary>(), tmp, t_);
				}
				reduced_to_full_aux(state_, full_size, reduced_size, reduced, tmp, full);
			}

			void update_quantities(const double t, const TVector &x);
			void solution_changed(const TVector &newX);

			void init_lagging(const TVector &x);
			void update_lagging(const TVector &x);

			void set_full_size(const bool val)
			{
				if (!val)
					reduced_size = actual_reduced_size;
				else
					reduced_size = full_size;
			}

		private:
			const State &state_;
			double t_;
			const assembler::RhsAssembler &rhs_assembler_;

			const int full_size, actual_reduced_size;
			int reduced_size;
			std::vector<std::shared_ptr<Form>> forms_;

			virtual void hessian_full(const TVector &x, THessian &hessian);
		};
	} // namespace solver
} // namespace polyfem
