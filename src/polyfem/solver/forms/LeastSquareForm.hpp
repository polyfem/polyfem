#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	class LeastSquareForm : public Form
	{
	public:
		LeastSquareForm(const StiffnessMatrix &mass): mass_(mass)
		{
			if (mass_.size() == 0)
				log_and_throw_error("mass matrix not available!");
		}

        void set_target(const Eigen::MatrixXd &target) { target_ = target; }
    
	protected:
		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value_unweighted(const Eigen::VectorXd &x) const override
        {
			if (mass_.rows() != x.size())
				log_and_throw_error("mass matrix size inconsistent!");
            if (target_.size() == x.size())
                return ((target_ - x).transpose() * mass_ * (target_ - x))(0);
            else
                return 0;
        }

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
        {
			if (mass_.rows() != x.size())
				log_and_throw_error("mass matrix size inconsistent!");
            if (target_.size() == x.size())
                gradv = mass_ * (x - target_);
            else
                gradv.setZero(mass_.rows());
        }

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override
        {
			if (mass_.rows() != x.size())
				log_and_throw_error("mass matrix size inconsistent!");
            if (target_.size() == x.size())
                hessian = 2 * mass_;
            else
			{
                hessian.resize(mass_.rows(), mass_.cols());
				hessian.setZero();
			}
        }

	private:
		const StiffnessMatrix &mass_;                                    ///< Mass matrix
		Eigen::MatrixXd target_;
	};
} // namespace polyfem::solver
