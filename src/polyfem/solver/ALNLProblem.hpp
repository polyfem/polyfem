#pragma once

#include "NLProblem.hpp"
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/State.hpp>

#include <cppoptlib/problem.h>

namespace polyfem
{
	namespace solver
	{
		class ALNLProblem : public NLProblem
		{
		public:
			typedef NLProblem super;
			using typename cppoptlib::Problem<double>::Scalar;
			using typename cppoptlib::Problem<double>::TVector;
			using typename super::THessian;

			ALNLProblem(const State &state, const assembler::RhsAssembler &rhs_assembler, const double t, const double dhat, const double weight);
			TVector initial_guess();
			void set_weight(const double w) { weight_ = w; }

			double value(const TVector &x) override { return super::value(x); }
			double value(const TVector &x, const bool only_elastic) override;
			void gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv, const bool only_elastic = false) override;
			void update_quantities(const double t, const TVector &x) override;
			void update_target(const double t);

			bool stop(const TVector &x) override;

#include <polyfem/utils/DisableWarnings.hpp>
			void hessian_full(const TVector &x, THessian &gradv) override;
#include <polyfem/utils/EnableWarnings.hpp>

		private:
			double weight_;
			double stop_dist_;
			THessian hessian_AL_;
			std::vector<int> not_boundary_;
			Eigen::MatrixXd target_x_; // actually a vector with the same size as x

			void compute_distance(const TVector &x, TVector &res);
		};
	} // namespace solver
} // namespace polyfem
