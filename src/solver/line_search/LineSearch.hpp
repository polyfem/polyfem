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

		virtual double line_search(
			const TVector &x,
			const TVector &grad,
			ProblemType &objFunc) = 0;

		static std::shared_ptr<LineSearch<ProblemType>> construct_line_search(const std::string &name);

		static void save_sampled_values(const std::string &filename,
										const TVector &x,
										const TVector &grad,
										ProblemType &objFunc,
										const double starting_step_size = 1e-1,
										const int num_samples = 1000);

		virtual void reset_times()
		{
			checking_for_nan_inf_time = 0;
			broad_phase_ccd_time = 0;
			ccd_time = 0;
			constrain_set_update_time = 0;
			classical_line_search_time = 0;
		}

		double checking_for_nan_inf_time;
		double broad_phase_ccd_time;
		double ccd_time;
		double constrain_set_update_time;
		double classical_line_search_time;
	};
} // namespace polyfem

#include "LineSearch.tpp"
