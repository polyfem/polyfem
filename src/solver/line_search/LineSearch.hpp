#pragma once

namespace polyfem
{
	template <typename ProblemType>
	class LineSearch
	{
	public:
		using Scalar = typename ProblemType::Scalar;
		using TVector = typename ProblemType::TVector;

		LineSearch() {}
		virtual ~LineSearch() = default;

		virtual double linesearch(
			const TVector &x,
			const TVector &grad,
			ProblemType &objFunc) = 0;

		static std::shared_ptr<LineSearch<ProblemType>> construct_line_search(const std::string &name);

		virtual void reset_times()
		{
			checking_for_nan_inf_time = 0;
			broad_phase_ccd_time = 0;
			ccd_time = 0;
			constrain_set_update_time = 0;
			classical_linesearch_time = 0;
		}

		double checking_for_nan_inf_time;
		double broad_phase_ccd_time;
		double ccd_time;
		double constrain_set_update_time;
		double classical_linesearch_time;
	};
} // namespace polyfem

#include "LineSearch.tpp"
