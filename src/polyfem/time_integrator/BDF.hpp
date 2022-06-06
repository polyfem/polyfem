#pragma once

#include <Eigen/Core>

#include <deque>

namespace polyfem
{
	namespace time_integrator
	{
		class BDF
		{
		public:
			BDF(int order);

			double alpha() const;
			void rhs(Eigen::VectorXd &rhs) const;

			void new_solution(Eigen::VectorXd &rhs);

		private:
			std::deque<Eigen::VectorXd> history_;
			int order_;
		};
	} // namespace time_integrator
} // namespace polyfem
