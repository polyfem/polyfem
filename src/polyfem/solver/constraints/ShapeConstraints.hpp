#include "Constraints.hpp"

namespace polyfem
{
	class ShapeConstraints : public Constraints
	{
	public:
		void update_state(std::shared_ptr<State> state, const Eigen::VectorXd &reduced) override
		{
			Eigen::VectorXd full;
			reduced_to_full(reduced, full);
			state->set_v(full);
		}

		ShapeConstraints(json constraint_params, int full_size, int reduced_size) : Constraints(constraint_params, full_size, reduced_size)
		{
			if (constraint_params["type"] == "identity")
			{
				full_to_reduced_ = [this](const DiffVector &full) {
					DiffVector reduced(reduced_size_);
					for (int i = 0; i < reduced_size_; ++i)
						reduced(i) = full(i);
					return reduced;
				};
				reduced_to_full_ = [this](const Eigen::VectorXd &reduced) {
					Eigen::VectorXd full(full_size_);
					for (int i = 0; i < full_size_; ++i)
						full(i) = reduced(i);
					return full;
				};
			}
		}
	};
} // namespace polyfem