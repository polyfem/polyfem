#pragma once

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/State.hpp>

#include <polyfem/utils/MatrixUtils.hpp>

#include <cppoptlib/problem.h>

namespace polyfem
{
	namespace solver
	{
		class NLHomogenizationProblem : public cppoptlib::Problem<double>
		{
		public:
			using typename cppoptlib::Problem<double>::Scalar;
			using typename cppoptlib::Problem<double>::TVector;
			typedef StiffnessMatrix THessian;

			NLHomogenizationProblem(State &state, const bool no_reduced = false);
			void init(const TVector &displacement) {}

			double value(const TVector &x) override;
			double target_value(const TVector &x) { return value(x); }
			void gradient(const TVector &x, TVector &gradv) override;
			void target_gradient(const TVector &x, TVector &gradv) { gradient(x, gradv); }
			void gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv, const bool only_elastic = false);

			double value(const TVector &x, const bool only_elastic);
			void gradient(const TVector &x, TVector &gradv, const bool only_elastic);

			void smoothing(const TVector &x, TVector &new_x) {}
			bool is_step_valid(const TVector &x0, const TVector &x1);
			TVector force_inequality_constraint(const TVector &x0, const TVector &dx) { return x0 + dx; }
			bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; }
			double max_step_size(const TVector &x0, const TVector &x1) { return 1; }
			bool is_intersection_free(const TVector &x) { return true; }

			int n_inequality_constraints() { return 0; }
			double inequality_constraint_val(const TVector &x, const int index) { assert(false); return std::nan(""); }
			TVector inequality_constraint_grad(const TVector &x, const int index) { assert(false); return TVector(); }

			TVector get_lower_bound(const TVector& x) 
			{
				TVector min(x.size());
				min.setConstant(std::numeric_limits<double>::min());
				return min; 
			}
			TVector get_upper_bound(const TVector& x) 
			{
				TVector max(x.size());
				max.setConstant(std::numeric_limits<double>::max());
				return max; 
			}

			void line_search_begin(const TVector &x0, const TVector &x1) {}
			void line_search_end(bool failed) {}
			void post_step(const int iter_num, const TVector &x) {}
			void save_to_file(const TVector &x0){};
			bool remesh(TVector &x) { return false; };

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

				Eigen::MatrixXd tmp = full;
				if (state.args["boundary_conditions"]["periodic_boundary"] && !state.args["space"]["advanced"]["periodic_basis"])
					state.full_to_periodic(tmp);
				
				long j = 0;
				size_t k = 0;
				for (int i = 0; i < tmp.size(); ++i)
				{
					if (k < state.boundary_nodes.size() && state.boundary_nodes[k] == i)
					{
						++k;
						continue;
					}

					reduced(j++) = tmp(i);
				}
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
				Eigen::MatrixXd tmp(reduced_size + state.boundary_nodes.size(), 1);
				for (int i = 0; i < tmp.size(); ++i)
				{
					if (k < state.boundary_nodes.size() && state.boundary_nodes[k] == i)
					{
						++k;
						tmp(i) = rhs(i);
						continue;
					}

					tmp(i) = reduced(j++);
				}

				if (state.args["boundary_conditions"]["periodic_boundary"] && !state.args["space"]["advanced"]["periodic_basis"])
					full = state.periodic_to_full(full_size, tmp);
				else
					full = tmp;
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

			void solution_changed(const TVector &newX) {}

			const Eigen::MatrixXd &current_rhs();

			bool stop(const TVector &x) { return false; }

			void set_project_to_psd(bool val) { project_to_psd = val; }
			bool is_project_to_psd() const { return project_to_psd; }

			inline double get_full_size() const { return full_size; }
			inline double get_reduced_size() const { return reduced_size; }

			void compute_cached_stiffness() {}

			void set_test_strain(const Eigen::MatrixXd &test_strain) { test_strain_ = test_strain; }

			utils::SpareMatrixCache mat_cache;

			StiffnessMatrix cached_stiffness;
			
		protected:
			State &state;

		private:
			const assembler::AssemblerUtils &assembler;
			Eigen::MatrixXd _current_rhs;

			const int full_size, reduced_size;
			bool rhs_computed;
			bool project_to_psd;

			Eigen::MatrixXd test_strain_;
		};
	} // namespace solver
} // namespace polyfem
