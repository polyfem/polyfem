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

		SolverWithBoxConstraints(const polyfem::json &solver_params, const double dt, const double characteristic_length)
			: Superclass(solver_params, dt, characteristic_length)
		{
			if (solver_params["max_change"].is_number())
				max_change_val_ = solver_params["max_change"];
			else
				nlohmann::adl_serializer<Eigen::VectorXd>::from_json(solver_params["max_change"], max_change_);

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
					else if (solver_params["bounds"][0].is_array())
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

		double compute_grad_norm(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const override
		{
			auto min = get_lower_bound(x, false);
			auto max = get_upper_bound(x, false);

			return ((x - grad).cwiseMax(min).cwiseMin(max) - x).norm();
			// Eigen::VectorXd proj_grad = grad;
			// for (int i = 0; i < x.size(); i++)
			// 	if (x(i) < min(i) + 1e-14 || x(i) > max(i) - 1e-14)
			// 		proj_grad(i) = 0;

			// return proj_grad.norm();
		}

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x, bool consider_max_change = true) const
		{
			Eigen::VectorXd min;
			if (bounds_.rows() == x.size())
				min = bounds_.col(0);
			else if (bounds_.size() == 2)
				min = Eigen::VectorXd::Constant(x.size(), 1, bounds_(0));
			else
				polyfem::log_and_throw_error("Invalid bounds!");

			if (consider_max_change)
				return min.array().max(x.array() - get_max_change(x).array());
			else
				return min;
		}
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x, bool consider_max_change = true) const
		{
			Eigen::VectorXd max;
			if (bounds_.rows() == x.size())
				max = bounds_.col(1);
			else if (bounds_.size() == 2)
				max = Eigen::VectorXd::Constant(x.size(), 1, bounds_(1));
			else
				polyfem::log_and_throw_error("Invalid bounds!");

			if (consider_max_change)
				return max.array().min(x.array() + get_max_change(x).array());
			else
				return max;
		}

		Eigen::VectorXd get_max_change(const Eigen::VectorXd &x) const
		{
			if (max_change_.size() == x.size())
				return max_change_;
			else if (max_change_val_ > 0)
				return Eigen::VectorXd::Ones(x.size()) * max_change_val_;
			else
				throw std::runtime_error("Invalid max change!");

			return Eigen::VectorXd();
		}

	private:
		Eigen::MatrixXd bounds_;

		double max_change_val_ = 0;
		Eigen::VectorXd max_change_;
	};

} // namespace cppoptlib