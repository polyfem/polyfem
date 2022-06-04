#include "BDF.hpp"

#include <vector>
#include <array>

namespace polyfem
{
	static const std::array<double, 6> alphas = {{1,
												  3. / 2.,
												  11. / 6.,
												  25. / 12.,
												  137. / 60.,
												  147. / 60.}};

	static const std::array<std::vector<double>, 6> weights =
		{{{1},
		  {-1. / 2., 2},
		  {1. / 3., -3. / 2., 3},
		  {-1. / 4., 4. / 3., -3, 4},
		  {1. / 5, -5. / 4., 10. / 3., -5, 5},
		  {-1. / 6., 6. / 5., -15. / 4., 20. / 3., -15. / 2., 6.}}};

	BDF::BDF(int order) : order_(order)
	{
		order_ = std::max(1, std::min(order_, 6));
	}

	double BDF::alpha() const
	{
		assert(history_.size() > 0);
		return alphas[history_.size() - 1];
	}

	void BDF::rhs(Eigen::VectorXd &rhs) const
	{
		assert(history_.size() > 0);
		rhs.resize(history_.front().size());
		rhs.setZero();
		const auto &w = weights[history_.size() - 1];
		for (int i = 0; i < history_.size(); ++i)
		{
			rhs += history_[i] * w[i];
		}
	}

	void BDF::new_solution(Eigen::VectorXd &rhs)
	{
		if (history_.size() >= order_)
		{
			history_.pop_front();
		}

		history_.push_back(rhs);
		assert(history_.size() <= order_);
	}
} // namespace polyfem
