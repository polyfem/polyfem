#include "CompositeForms.hpp"
#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem::solver
{
    namespace {
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}

		double dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }

		typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> Diff;

		template <typename T>
		T homo_stress_aux(const Eigen::Matrix<T, Eigen::Dynamic, 1> &F)
		{
			T val1 = F(0) * F(0) + F(1) * F(1) + F(2) * F(2);
			T val2 = F(3) * F(3);

			return sqrt(val1 / (val2 + val1));
		}

		Eigen::VectorXd homo_stress_aux_grad(const Eigen::VectorXd &F)
		{
			DiffScalarBase::setVariableCount(F.size());
			Eigen::Matrix<Diff, Eigen::Dynamic, 1> full_diff(F.size());
			for (int i = 0; i < F.size(); i++)
				full_diff(i) = Diff(i, F(i));
			auto reduced_diff = homo_stress_aux(full_diff);

			Eigen::VectorXd grad(F.size());
			for (int i = 0; i < F.size(); ++i)
				grad(i) = reduced_diff.getGradient()(i);

			return grad;
		}
    }

    double HomoCompositeForm::compose(const Eigen::VectorXd &inputs) const
    {
        if (inputs.size() != 4)
            throw std::runtime_error("Invalid input size for HomoCompositeForm!");
        return homo_stress_aux(inputs);
    }

    Eigen::VectorXd HomoCompositeForm::compose_grad(const Eigen::VectorXd &inputs) const
    {
        return homo_stress_aux_grad(inputs);
    }

	InequalityConstraintForm::InequalityConstraintForm(const std::vector<std::shared_ptr<AdjointForm>> &forms, const Eigen::Vector2d &bounds, const double power) : CompositeForm(forms), power_(power), bounds_(bounds)
	{
		assert(bounds_(1) >= bounds_(0));
	}

    double InequalityConstraintForm::compose(const Eigen::VectorXd &inputs) const
    {
        if (inputs.size() != 1)
            throw std::runtime_error("Invalid input size for InequalityConstraintForm!");
        
		return pow(std::max(bounds_(0) - inputs(0), 0.0), power_) + pow(std::max(inputs(0) - bounds_(1), 0.0), power_);
    }

    Eigen::VectorXd InequalityConstraintForm::compose_grad(const Eigen::VectorXd &inputs) const
    {
        Eigen::VectorXd grad(1);
		grad(0) = -power_ * pow(std::max(bounds_(0) - inputs(0), 0.0), power_ - 1) + power_ * pow(std::max(inputs(0) - bounds_(1), 0.0), power_ - 1);
		return grad;
    }
}