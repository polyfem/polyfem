#pragma once

#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem
{
	typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> DiffScalar;
	typedef Eigen::Matrix<DiffScalar, Eigen::Dynamic, 1> DiffVector;
	typedef Eigen::Matrix<DiffScalar, Eigen::Dynamic, Eigen::Dynamic> DiffMatrix;

	class Constraints
	{
	public:
		Constraints(const json &constraint_params, int full_size, int reduced_size) : constraint_params_(constraint_params), full_size_(full_size), reduced_size_(reduced_size) {}
		virtual ~Constraints() = default;

		virtual void update_state(std::shared_ptr<State> state, const Eigen::VectorXd &reduced) = 0;

		virtual int get_optimization_dim() { return reduced_size_; }

	protected:
		int full_size_;
		int reduced_size_;

		json constraint_params_;
	};
} // namespace polyfem