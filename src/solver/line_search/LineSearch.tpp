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
} // namespace polyfem
