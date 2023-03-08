#pragma once

#include "NonlinearSolver.hpp"
#include <polyfem/utils/MatrixUtils.hpp>
#include <filesystem>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/JSONUtils.hpp>

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
			max_change_ = solver_params["max_change"];
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
				else if (solver_params["bounds"].is_array() && solver_params["bounds"].size() == 2)
				{
					if (solver_params["bounds"][0].is_number())
					{
						bounds_.setZero(1, 2);
						bounds_ << solver_params["bounds"][0], solver_params["bounds"][1];
					}
					else if (solver_params["bounds"][0].is_array() > 0)
					{
						bounds_.setZero(solver_params["bounds"][0].size(), 2);
						Eigen::VectorXd tmp;
						nlohmann::adl_serializer<Eigen::VectorXd>::from_json(solver_params["bounds"][0], tmp);
						bounds_.col(0) = tmp;
						nlohmann::adl_serializer<Eigen::VectorXd>::from_json(solver_params["bounds"][1], tmp);
						bounds_.col(1) = tmp;
					}
				}
			}
		}

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd min;
			if (bounds_.rows() == x.size())
				min = bounds_.col(0);
			else if (bounds_.size() == 2)
				min = Eigen::VectorXd::Constant(x.size(), 1, bounds_(0));
			else
				polyfem::log_and_throw_error("Invalid bounds!");
			
			return min.array().max(x.array() - max_change_);
		}
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd max;
			if (bounds_.rows() == x.size())
				max = bounds_.col(1);
			else if (bounds_.size() == 2)
				max = Eigen::VectorXd::Constant(x.size(), 1, bounds_(1));
			else
				polyfem::log_and_throw_error("Invalid bounds!");
			
			return max.array().min(x.array() + max_change_);
		}

	private:
		Eigen::MatrixXd bounds_;
		double max_change_;
	};

}