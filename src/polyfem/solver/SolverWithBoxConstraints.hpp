#pragma once

#include "NonlinearSolver.hpp"
#include <polyfem/utils/MatrixUtils.hpp>
#include <filesystem>
#include <polyfem/io/MatrixIO.hpp>

namespace cppoptlib
{
	template <typename ProblemType>
	class SolverWithBoxConstraints : public NonlinearSolver<ProblemType>
    {
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		SolverWithBoxConstraints(const polyfem::json &solver_params, const double dt)
			: Superclass(solver_params, dt)
		{
			if (solver_params.contains("bounds"))
			{
				if (solver_params["bounds"].is_string())
				{
					if (std::filesystem::is_regular_file(solver_params["bounds"].get<std::string>()))
					{
						polyfem::io::read_matrix(solver_params["bounds"].get<std::string>(), bounds_);
						assert(bounds_.cols() == 2);
					}
				}
				else if (solver_params["bounds"].is_array())
				{
					assert(solver_params["bounds"].size() == 2);
					bounds_.setZero(1, 2);
					bounds_ << solver_params["bounds"][0], solver_params["bounds"][1];
				}
			}
		}

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const
		{
			if (bounds_.rows() == x.size())
				return bounds_.col(0);
			else if (bounds_.size() == 2)
				return Eigen::VectorXd::Constant(x.size(), 1, bounds_(0));
			else
			{
				polyfem::log_and_throw_error("Invalid bounds!");
				return Eigen::VectorXd();
			}
		}
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const
		{
			if (bounds_.rows() == x.size())
				return bounds_.col(1);
			else if (bounds_.size() == 2)
				return Eigen::VectorXd::Constant(x.size(), 1, bounds_(1));
			else
			{
				polyfem::log_and_throw_error("Invalid bounds!");
				return Eigen::VectorXd();
			}
		}

	private:
		Eigen::MatrixXd bounds_;
	};

}