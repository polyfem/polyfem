#pragma once

#include "Form.hpp"

#include <polyfem/State.hpp>
#include <polyfem/utils/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/broad_phase/broad_phase.hpp>

namespace polyfem
{
	namespace solver
	{
		class ContactForm : public Form
		{
		public:
			ContactForm(const State &state,
						const double dhat,
						const bool use_adaptive_barrier_stiffness,
						const double &barrier_stiffness,
						const bool is_time_dependent,
						const ipc::BroadPhaseMethod broad_phase_method,
						const double ccd_tolerance,
						const int ccd_max_iterations);

			void init(const Eigen::VectorXd &x) override;

			double value(const Eigen::VectorXd &x) override;
			void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
			void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

			double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

			void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
			void line_search_end() override;

			void solution_changed(const Eigen::VectorXd &newX) override;
			void post_step(const int iter_num, const Eigen::VectorXd &x) override;

			void update_quantities(const double t, const Eigen::VectorXd &x) override;

		private:
			const double dhat_;
			const bool use_adaptive_barrier_stiffness_;
			const double &barrier_stiffness_;

			const bool is_time_dependent_;

			const ipc::BroadPhaseMethod broad_phase_method_;
			const double ccd_tolerance_;
			const int ccd_max_iterations_;

			double prev_distance_;
			double max_barrier_stiffness_;
			ipc::Constraints constraint_set_;
			ipc::Candidates candidates_;
			bool use_cached_candidates_ = false;

			const State &state_;

			void update_barrier_stiffness(const Eigen::VectorXd &x);
			void update_constraint_set(const Eigen::MatrixXd &displaced_surface);
		};
	} // namespace solver
} // namespace polyfem
