#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace polyfem::solver
{
	/// @brief A function f : x -> y.
	class Parametrization
	{
	public:
		virtual ~Parametrization() = default;

		/// @brief Eval x = f^-1 (y).
		///
		/// This is not a strict inverse in mathematical sense,
		/// one may choose "reasonable" x even if f is not one-to-one.
		///
		/// @param[in] y y.
		/// @return x.
		/// @throws std::runtime_error Throw if not implemented.
		virtual Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y);

		/// @brief Compute DOF of y given DOF of x.
		/// @param[in] x_size The DOF of x.
		/// @return DOF of y.
		virtual int size(const int x_size) const = 0;

		/// @brief Eval y = f(x).
		/// @param[in] x x.
		/// @return y.
		virtual Eigen::VectorXd eval(const Eigen::VectorXd &x) const = 0;

		/// @brief Apply jacobian for chain rule.
		///
		/// Let g(y) = g(f(x)).
		/// Given ∂g/∂y, compute ∂g/∂x = ∂g/∂y * ∂y/∂x.
		///
		/// @param[in] grad_full ∂g/∂y.
		/// @param[in] x Where ∂g/∂x is evaluated.
		/// @return ∂g/∂x.
		virtual Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const = 0;
	};

	class CompositeParametrization : public Parametrization
	{
	public:
		CompositeParametrization() = default;
		CompositeParametrization(std::vector<std::shared_ptr<Parametrization>> parametrizations) : parametrizations_(std::move(parametrizations)) {}

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;

		int size(const int x_size) const override;
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const override;

	private:
		const std::vector<std::shared_ptr<Parametrization>> parametrizations_;
	};
} // namespace polyfem::solver
