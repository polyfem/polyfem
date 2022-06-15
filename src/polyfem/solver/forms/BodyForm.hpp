#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/State.hpp>

namespace polyfem
{
	namespace solver
	{
		class BodyForm : public Form
		{
		public:
			BodyForm(const State &state, const assembler::RhsAssembler &rhs_assembler);

			double value(const Eigen::VectorXd &x) override;
			void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;

			void update_quantities(const double t, const Eigen::VectorXd &x) override;

		private:
			bool rhs_computed_;
			double t_;
			const State &state_;
			const assembler::RhsAssembler &rhs_assembler_;
			const assembler::AssemblerUtils &assembler_;

			Eigen::MatrixXd current_rhs_;

			const Eigen::MatrixXd &current_rhs();
		};
	} // namespace solver
} // namespace polyfem
