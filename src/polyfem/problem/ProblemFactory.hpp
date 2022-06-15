#pragma once

#include <polyfem/assembler/Problem.hpp>

#include <functional>
#include <memory>

namespace polyfem
{
	namespace problem
	{

		class ProblemFactory
		{
		public:
			static const ProblemFactory &factory();

			std::shared_ptr<assembler::Problem> get_problem(const std::string &problem) const;
			inline const std::vector<std::string> &get_problem_names() const { return problem_names_; }

		private:
			ProblemFactory();
			std::map<std::string, std::function<std::shared_ptr<assembler::Problem>()>> problems_;
			std::vector<std::string> problem_names_;
		};
	} // namespace problem
} // namespace polyfem
