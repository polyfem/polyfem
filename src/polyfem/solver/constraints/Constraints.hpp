#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem
{
	typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> DiffScalar;
	typedef Eigen::Matrix<DiffScalar, Eigen::Dynamic, 1> DiffVector;

	class Constraints
	{
	public:
		Constraints(json constraint_params, int full_size, int reduced_size) : constraint_params_(constraint_params), full_size_(full_size), reduced_size_(reduced_size) {}
		virtual ~Constraints() = default;

		virtual void full_to_reduced(const Eigen::VectorXd &full, Eigen::VectorXd &reduced, Eigen::MatrixXd &grad)
		{
			if (dreduced_dfull_ != nullptr)
			{
				reduced = full_to_reduced_(full);
				grad = dreduced_dfull_(full);
				return;
			}

			DiffScalarBase::setVariableCount(full_size_); // size of input vector
			assert(full.size() == full_size_);
			DiffVector full_diff(full_size_);
			for (int i = 0; i < full.size(); ++i)
				full_diff(i) = DiffScalar(i, full(i));
			auto reduced_diff = full_to_reduced_diff_(full_diff);
			assert(reduced_diff.size() == reduced_size_);
			reduced.resize(reduced_size_);
			grad.resize(reduced_size_, full_size_);
			for (int i = 0; i < reduced_size_; ++i)
			{
				reduced(i) = reduced_diff(i).getValue();
				for (int j = 0; j < full.size(); ++j)
					grad(i, j) = reduced_diff(i).getGradient()(j);
			}
		}

		virtual void reduced_to_full(const Eigen::VectorXd &reduced, Eigen::VectorXd &full)
		{
			full = reduced_to_full_(reduced);
		}

		virtual void update_state(std::shared_ptr<State> state, const Eigen::VectorXd &reduced) = 0;

	protected:
		// For differentiability, either define the function with differentiable types.
		std::function<DiffVector(const DiffVector &)> full_to_reduced_diff_;
		// Or explicitly define the function and gradient with Eigen types.
		std::function<Eigen::VectorXd(const Eigen::VectorXd &)> full_to_reduced_;
		std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> dreduced_dfull_;

		// Must define the inverse function with Eigen types, differentiability is not needed.
		std::function<Eigen::VectorXd(const Eigen::VectorXd &)> reduced_to_full_;

		int full_size_;
		int reduced_size_;

		json constraint_params_;
	};
} // namespace polyfem