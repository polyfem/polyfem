#pragma once

#include <polyfem/LineSearch.hpp>
#include <polyfem/ArmijoLineSearch.hpp>
#include <polyfem/BisectionLineSearch.hpp>
#include <polyfem/CppOptArmijoLineSearch.hpp>
#include <polyfem/MoreThuenteLineSearch.hpp>

namespace polyfem
{
	template <typename ProblemType>
	std::shared_ptr<LineSearch<ProblemType>> LineSearch<ProblemType>::construct_line_search(const std::string &name)
	{
		if (name == "armijo" || name == "Armijo")
		{
			return std::make_shared<ArmijoLineSearch<ProblemType>>();
		}
		else if (name == "armijo_alt" || name == "ArmijoAlt")
		{
			return std::make_shared<CppOptArmijoLineSearch<ProblemType>>();
		}
		else if (name == "bisection" || name == "Bisection")
		{
			return std::make_shared<BisectionLineSearch<ProblemType>>();
		}
		else if (name == "more_thuente" || name == "MoreThuente")
		{
			return std::make_shared<MoreThuenteLineSearch<ProblemType>>();
		}
		else if (name == "none")
		{
			return nullptr;
		}
		else
		{
			std::string msg = fmt::format("Unknown line search {}!", name);
			logger().error(msg);
			throw std::invalid_argument(msg);
		}
	}

	template <typename ProblemType>
	void LineSearch<ProblemType>::save_sampled_values(
		const std::string &filename,
		const typename ProblemType::TVector &x,
		const typename ProblemType::TVector &delta_x,
		ProblemType &objFunc,
		const double starting_step_size,
		const int num_samples)
	{
		std::ofstream samples(filename, std::ios::out);
		if (!samples.is_open())
		{
			spdlog::error("Unable to save sampled values to file \"{}\" !", filename);
			return;
		}

		samples << "alpha,f(x + alpha * delta_x),valid,decrease\n";

		objFunc.solution_changed(x);
		double fx = objFunc.value(x);

		Eigen::VectorXd alphas = Eigen::VectorXd::LinSpaced(2 * num_samples - 1, -starting_step_size, starting_step_size);
		for (int i = 0; i < alphas.size(); i++)
		{
			typename ProblemType::TVector new_x = x + alphas[i] * delta_x;
			objFunc.solution_changed(new_x);
			double fxi = objFunc.value(new_x);
			samples << alphas[i] << ","
					<< fxi << ","
					<< (objFunc.is_step_valid(x, new_x) ? "true" : "false") << ","
					<< (fxi < fx ? "true" : "false") << "\n";
		}

		samples.close();
	}
} // namespace polyfem
