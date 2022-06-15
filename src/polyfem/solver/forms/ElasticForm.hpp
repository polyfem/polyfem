#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/State.hpp>

namespace polyfem
{
	namespace solver
	{
		class ElasticForm : public Form
		{
		public:
			ElasticForm(const State &state);

			double value(const Eigen::VectorXd &x) override;
			void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
			void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

			bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		private:
			const State &state_;
			const assembler::AssemblerUtils &assembler_;
			const std::string formulation_;
			StiffnessMatrix cached_stiffness_;
			utils::SpareMatrixCache mat_cache_;

			void compute_cached_stiffness();
		};
	} // namespace solver
} // namespace polyfem
