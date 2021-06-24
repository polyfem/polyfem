#pragma once

#include <polyfem/NLProblem.hpp>
#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/RhsAssembler.hpp>
#include <polyfem/State.hpp>

#include <cppoptlib/problem.h>

namespace polyfem
{
	class ALNLProblem : public NLProblem
	{
	public:
		typedef NLProblem super;
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;
		using typename super::THessian;

		ALNLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool project_to_psd, const double weight);
		TVector initial_guess();
		void set_weight(const double w) { weight_ = w; }

		double value(const TVector &x) override { return super::value(x); }
		double value(const TVector &x, const bool only_elastic) override;
		void gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv, const bool only_elastic = false) override;
		void update_quantities(const double t, const TVector &x) override;

		bool stop(const TVector &x) override;

#include <polyfem/DisableWarnings.hpp>
		void hessian_full(const TVector &x, THessian &gradv) override;
#include <polyfem/EnableWarnings.hpp>

	private:
		double weight_;
		double stop_dist_;
		THessian hessian_;
		std::vector<int> not_boundary_;
		Eigen::MatrixXd displaced_;

		void compute_distance(const TVector &x, TVector &res);
	};
} // namespace polyfem
