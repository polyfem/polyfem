#pragma once

#include "polyfem/utils/CudaBoth.hpp"

#include <Eigen/Core>
#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/std/cmath>
#else
#include <cmath>
#endif

namespace polyfem::autodiff
{

// If cuda is enabled, use math function in cuda::std.
// Else use math function from std namespace.
#ifdef POLYFEM_WITH_CUDA
	namespace math = cuda::std;
#else
	namespace math = std;
#endif

	/// @brief Auto diff scalar with first order derivative.
	///
	/// Given unknown u = (u0, u1, u2, ...)  where dim = value_dim,
	/// Compute gradient df/du_i
	///
	/// @tparam Scalar Floating point type.
	/// @tparam value_dim Dimension of the unknown u.
	template <typename Scalar, int value_dim>
	struct ScalarGrad
	{
	public:
		using Grad = Eigen::Matrix<Scalar, value_dim, 1>;
		static constexpr int VALUE_DIM = value_dim;

	private:
		Scalar val_;
		Grad grad_;

	public:
		// ======================================================================
		/// Constructors and accessors
		// ======================================================================

		/// Create a new constant automatic differentiation scalar
		POLYFEM_BOTH explicit ScalarGrad(Scalar val = 0)
			: val_(val), grad_(Grad::Zero()) {}

		/// Construct a new scalar with the specified value and one first derivative
		/// set to 1
		POLYFEM_BOTH ScalarGrad(Scalar val, int index)
			: val_(val), grad_(Grad::Zero())
		{
			assert(index >= 0 && index < VALUE_DIM);
			grad_(index) = Scalar{1};
		}

		/// Construct a scalar associated with the given gradient
		POLYFEM_BOTH ScalarGrad(Scalar val, Grad grad) : val_(val), grad_(grad) {}

		POLYFEM_BOTH const Scalar &get_val() const { return val_; }
		POLYFEM_BOTH const Grad &get_grad() const { return grad_; }

		// ======================================================================
		/// Addition
		// ======================================================================

		POLYFEM_BOTH friend ScalarGrad operator+(const ScalarGrad &lhs,
												 const ScalarGrad &rhs)
		{
			return ScalarGrad(lhs.val_ + rhs.val_, lhs.grad_ + rhs.grad_);
		}

		POLYFEM_BOTH friend ScalarGrad operator+(const ScalarGrad &lhs,
												 const Scalar &rhs)
		{
			return ScalarGrad(lhs.val_ + rhs, lhs.grad_);
		}

		POLYFEM_BOTH friend ScalarGrad operator+(const Scalar &lhs,
												 const ScalarGrad &rhs)
		{
			return ScalarGrad(rhs.val_ + lhs, rhs.grad_);
		}

		POLYFEM_BOTH ScalarGrad &operator+=(const ScalarGrad &s)
		{
			*this = *this + s;
			return *this;
		}

		POLYFEM_BOTH ScalarGrad &operator+=(const Scalar &s)
		{
			*this = *this + s;
			return *this;
		}

		// ======================================================================
		/// Subtraction
		// ======================================================================

		POLYFEM_BOTH friend ScalarGrad operator-(const ScalarGrad &lhs,
												 const ScalarGrad &rhs)
		{
			return ScalarGrad(lhs.val_ - rhs.val_, lhs.grad_ - rhs.grad_);
		}

		POLYFEM_BOTH friend ScalarGrad operator-(const ScalarGrad &lhs,
												 const Scalar &rhs)
		{
			return ScalarGrad(lhs.val_ - rhs, lhs.grad_);
		}

		POLYFEM_BOTH friend ScalarGrad operator-(const Scalar &lhs,
												 const ScalarGrad &rhs)
		{
			return ScalarGrad(lhs - rhs.val_, -rhs.grad_);
		}

		POLYFEM_BOTH friend ScalarGrad operator-(const ScalarGrad &s)
		{
			return ScalarGrad(-s.val_, -s.grad_);
		}

		POLYFEM_BOTH ScalarGrad &operator-=(const ScalarGrad &s)
		{
			*this = *this - s;
			return *this;
		}

		POLYFEM_BOTH ScalarGrad &operator-=(const Scalar &s)
		{
			*this = *this - s;
			return *this;
		}

		// ======================================================================
		/// Division
		// ======================================================================

		POLYFEM_BOTH friend ScalarGrad operator/(const ScalarGrad &lhs,
												 const Scalar &rhs)
		{
			assert(rhs != 0);
			Scalar inv = Scalar{1} / rhs;
			return ScalarGrad(lhs.val_ * inv, lhs.grad_ * inv);
		}

		POLYFEM_BOTH friend ScalarGrad operator/(const Scalar &lhs,
												 const ScalarGrad &rhs)
		{
			return lhs * inverse(rhs);
		}

		POLYFEM_BOTH friend ScalarGrad operator/(const ScalarGrad &lhs,
												 const ScalarGrad &rhs)
		{
			return lhs * inverse(rhs);
		}

		POLYFEM_BOTH friend ScalarGrad inverse(const ScalarGrad &s)
		{
			assert(s.val_ != Scalar{0});
			// vn = 1/v, Dvn = -1/(v^2) Dv
			return ScalarGrad(Scalar{1} / s.val_, s.grad_ / -(s.val_ * s.val_));
		}

		POLYFEM_BOTH ScalarGrad &operator/=(const ScalarGrad &s)
		{
			*this = *this / s;
			return *this;
		}

		POLYFEM_BOTH ScalarGrad &operator/=(const Scalar &s)
		{
			*this = *this / s;
			return *this;
		}

		// ======================================================================
		/// Multiplication
		// ======================================================================

		POLYFEM_BOTH friend ScalarGrad operator*(const ScalarGrad &lhs,
												 const Scalar &rhs)
		{
			return ScalarGrad(lhs.val_ * rhs, lhs.grad_ * rhs);
		}

		POLYFEM_BOTH friend ScalarGrad operator*(const Scalar &lhs,
												 const ScalarGrad &rhs)
		{
			return ScalarGrad(rhs.val_ * lhs, rhs.grad_ * lhs);
		}

		POLYFEM_BOTH friend ScalarGrad operator*(const ScalarGrad &lhs,
												 const ScalarGrad &rhs)
		{
			// Product rule
			return ScalarGrad(lhs.val_ * rhs.val_,
							  rhs.grad_ * lhs.val_ + lhs.grad_ * rhs.val_);
		}

		POLYFEM_BOTH ScalarGrad &operator*=(const ScalarGrad &s)
		{
			*this = *this * s;
			return *this;
		}

		POLYFEM_BOTH ScalarGrad &operator*=(const Scalar &s)
		{
			*this = *this * s;
			return *this;
		}

		// ======================================================================
		/// Miscellaneous functions
		// ======================================================================

		POLYFEM_BOTH friend ScalarGrad sqrt(const ScalarGrad &s)
		{
			assert(s.val_ > Scalar{0});
			Scalar sqrtVal = math::sqrt(s.val_);
			Scalar temp = Scalar{1} / (Scalar{2} * sqrtVal);
			// vn = sqrt(v)
			// Dvn = 1/(2 sqrt(v)) Dv
			return ScalarGrad(sqrtVal, s.grad_ * temp);
		}

		POLYFEM_BOTH friend ScalarGrad pow(const ScalarGrad &s, const Scalar &a)
		{
			if (a == Scalar{0})
			{
				return ScalarGrad(Scalar{1});
			}
			if (a == Scalar{1})
			{
				return s;
			}

			assert(s.val_ > Scalar{0});
			Scalar powVal = math::pow(s.val_, a);
			Scalar temp = a * math::pow(s.val_, a - 1);
			// vn = v ^ a, Dvn = a*v^(a-1) * Dv
			return ScalarGrad(powVal, s.grad_ * temp);
		}

		POLYFEM_BOTH friend ScalarGrad exp(const ScalarGrad &s)
		{
			Scalar expVal = math::exp(s.val_);
			// vn = exp(v), Dvn = exp(v) * Dv
			return ScalarGrad(expVal, s.grad_ * expVal);
		}

		POLYFEM_BOTH friend ScalarGrad log(const ScalarGrad &s)
		{
			assert(s.val_ > Scalar{0});
			Scalar logVal = math::log(s.val_);
			// vn = log(v), Dvn = Dv / v
			return ScalarGrad(logVal, s.grad_ / s.val_);
		}

		POLYFEM_BOTH friend ScalarGrad sin(const ScalarGrad &s)
		{
			// vn = sin(v), Dvn = cos(v) * Dv
			return ScalarGrad(math::sin(s.val_), s.grad_ * math::cos(s.val_));
		}

		POLYFEM_BOTH friend ScalarGrad cos(const ScalarGrad &s)
		{
			// vn = cos(v), Dvn = -sin(v) * Dv
			return ScalarGrad(math::cos(s.val_), s.grad_ * -math::sin(s.val_));
		}

		POLYFEM_BOTH friend ScalarGrad acos(const ScalarGrad &s)
		{
			assert(math::abs(s.val_) < 1);

			Scalar temp = -math::sqrt(Scalar{1} - s.val_ * s.val_);
			// vn = acos(v), Dvn = -1/sqrt(1-v^2) * Dv
			return ScalarGrad(math::acos(s.val_), s.grad_ * (Scalar{1} / temp));
		}

		POLYFEM_BOTH friend ScalarGrad asin(const ScalarGrad &s)
		{
			assert(math::abs(s.val_) < 1);

			Scalar temp = math::sqrt(Scalar{1} - s.val_ * s.val_);
			// vn = asin(v), Dvn = 1/sqrt(1-v^2) * Dv
			return ScalarGrad(math::asin(s.val_), s.grad_ * (Scalar{1} / temp));
		}

		POLYFEM_BOTH friend ScalarGrad atan2(const ScalarGrad &y,
											 const ScalarGrad &x)
		{
			Scalar denom = x.val_ * x.val_ + y.val_ * y.val_;
			assert(denom != Scalar{0});
			// vn = atan2(y, x), Dvn = (x*Dy - y*Dx) / (x^2 + y^2)
			return ScalarGrad(math::atan2(y.val_, x.val_),
							  y.grad_ * (x.val_ / denom) - x.grad_ * (y.val_ / denom));
		}

		// ======================================================================
		/// Comparison
		// ======================================================================

		POLYFEM_BOTH friend bool operator<(const ScalarGrad &lhs, const Scalar &rhs)
		{
			return lhs.val_ < rhs;
		}

		POLYFEM_BOTH friend bool operator<(const Scalar &lhs, const ScalarGrad &rhs)
		{
			return lhs < rhs.val_;
		}

		POLYFEM_BOTH friend bool operator<(const ScalarGrad &lhs,
										   const ScalarGrad &rhs)
		{
			return lhs.val_ < rhs.val_;
		}

		POLYFEM_BOTH friend bool operator<=(const ScalarGrad &lhs,
											const Scalar &rhs)
		{
			return lhs.val_ <= rhs;
		}

		POLYFEM_BOTH friend bool operator<=(const Scalar &lhs,
											const ScalarGrad &rhs)
		{
			return lhs <= rhs.val_;
		}

		POLYFEM_BOTH friend bool operator<=(const ScalarGrad &lhs,
											const ScalarGrad &rhs)
		{
			return lhs.val_ <= rhs.val_;
		}

		POLYFEM_BOTH friend bool operator>(const ScalarGrad &lhs, const Scalar &rhs)
		{
			return lhs.val_ > rhs;
		}

		POLYFEM_BOTH friend bool operator>(const Scalar &lhs, const ScalarGrad &rhs)
		{
			return lhs > rhs.val_;
		}

		POLYFEM_BOTH friend bool operator>(const ScalarGrad &lhs,
										   const ScalarGrad &rhs)
		{
			return lhs.val_ > rhs.val_;
		}

		POLYFEM_BOTH friend bool operator>=(const ScalarGrad &lhs,
											const Scalar &rhs)
		{
			return lhs.val_ >= rhs;
		}

		POLYFEM_BOTH friend bool operator>=(const Scalar &lhs,
											const ScalarGrad &rhs)
		{
			return lhs >= rhs.val_;
		}

		POLYFEM_BOTH friend bool operator>=(const ScalarGrad &lhs,
											const ScalarGrad &rhs)
		{
			return lhs.val_ >= rhs.val_;
		}

		POLYFEM_BOTH friend bool operator==(const ScalarGrad &lhs,
											const Scalar &rhs)
		{
			return lhs.val_ == rhs;
		}

		POLYFEM_BOTH friend bool operator==(const Scalar &lhs,
											const ScalarGrad &rhs)
		{
			return lhs == rhs.val_;
		}

		POLYFEM_BOTH friend bool operator==(const ScalarGrad &lhs,
											const ScalarGrad &rhs)
		{
			return lhs.val_ == rhs.val_;
		}

		POLYFEM_BOTH friend bool operator!=(const ScalarGrad &lhs,
											const Scalar &rhs)
		{
			return lhs.val_ != rhs;
		}

		POLYFEM_BOTH friend bool operator!=(const Scalar &lhs,
											const ScalarGrad &rhs)
		{
			return lhs != rhs.val_;
		}

		POLYFEM_BOTH friend bool operator!=(const ScalarGrad &lhs,
											const ScalarGrad &rhs)
		{
			return lhs.val_ != rhs.val_;
		}

		POLYFEM_BOTH friend ScalarGrad abs(const ScalarGrad &s)
		{
			return s.val_ < Scalar{0} ? -s : s;
		}

		POLYFEM_BOTH friend ScalarGrad min(const ScalarGrad &lhs,
										   const ScalarGrad &rhs)
		{
			return rhs.val_ < lhs.val_ ? rhs : lhs;
		}

		POLYFEM_BOTH friend ScalarGrad min(const ScalarGrad &lhs, const Scalar &rhs)
		{
			return rhs < lhs.val_ ? ScalarGrad(rhs) : lhs;
		}

		POLYFEM_BOTH friend ScalarGrad min(const Scalar &lhs, const ScalarGrad &rhs)
		{
			return rhs.val_ < lhs ? rhs : ScalarGrad(lhs);
		}

		POLYFEM_BOTH friend ScalarGrad max(const ScalarGrad &lhs,
										   const ScalarGrad &rhs)
		{
			return lhs.val_ < rhs.val_ ? rhs : lhs;
		}

		POLYFEM_BOTH friend ScalarGrad max(const ScalarGrad &lhs, const Scalar &rhs)
		{
			return lhs.val_ < rhs ? ScalarGrad(rhs) : lhs;
		}

		POLYFEM_BOTH friend ScalarGrad max(const Scalar &lhs, const ScalarGrad &rhs)
		{
			return lhs < rhs.val_ ? rhs : ScalarGrad(lhs);
		}
	};

	/// @brief Auto diff scalar with upper right block of second order derivative.
	///
	/// Given unknown a = (a0, a1, ...) and b = (b0, b1, ...) where dim = value_dim,
	/// Compute:
	/// - gradient df/da and df/db.
	/// - hessian (d^2 f)/(da_i db_j)
	///
	/// @tparam Scalar Floating point type.
	/// @tparam value_dim Dimension of the unknown a and b.
	template <typename Scalar, int value_dim>
	class ScalarHess
	{
	public:
		static constexpr int VALUE_DIM = value_dim;
		using Grad = Eigen::Matrix<Scalar, 2 * value_dim, 1>;
		using Hess = Eigen::Matrix<Scalar, value_dim, value_dim, Eigen::RowMajor>;

	private:
		Scalar val_;
		/// Gradient. Layout: [a0 a1 ... b0 b1 ...]
		Grad grad_;
		/// Upper right block of hessian. Layout:
		/// [ df/(da0 db0) df/(da0 db1) ... ]
		/// [ df/(da1 db0) df/(da1 db1) ... ]
		/// [ ...          ...          ... ]
		Hess hess_;

	public:
		// ======================================================================
		/// Constructors and accessors
		// ======================================================================

		/// Create a new constant automatic differentiation scalar
		POLYFEM_BOTH explicit ScalarHess(Scalar val = 0)
			: val_(val), grad_(Grad::Zero()), hess_(Hess::Zero()) {}

		/// Construct a new scalar with the specified value and one first derivative
		/// set to 1
		POLYFEM_BOTH ScalarHess(Scalar val, int index)
			: val_(val), grad_(Grad::Zero()), hess_(Hess::Zero())
		{
			assert(index >= 0 && index < 2 * VALUE_DIM);
			grad_(index) = Scalar{1};
		}

		/// Construct a scalar associated with the given gradient and Hessian
		POLYFEM_BOTH ScalarHess(Scalar val, Grad grad, Hess hess)
			: val_(val), grad_(grad), hess_(hess) {}

		POLYFEM_BOTH const Scalar &get_val() const { return val_; }
		POLYFEM_BOTH const Grad &get_grad() const { return grad_; }
		POLYFEM_BOTH const Hess &get_hess() const { return hess_; }

		// ======================================================================
		/// Addition
		// ======================================================================

		POLYFEM_BOTH friend ScalarHess operator+(const ScalarHess &lhs,
												 const ScalarHess &rhs)
		{
			return ScalarHess(lhs.val_ + rhs.val_, lhs.grad_ + rhs.grad_,
							  lhs.hess_ + rhs.hess_);
		}

		POLYFEM_BOTH friend ScalarHess operator+(const ScalarHess &lhs,
												 const Scalar &rhs)
		{
			return ScalarHess(lhs.val_ + rhs, lhs.grad_, lhs.hess_);
		}

		POLYFEM_BOTH friend ScalarHess operator+(const Scalar &lhs,
												 const ScalarHess &rhs)
		{
			return ScalarHess(rhs.val_ + lhs, rhs.grad_, rhs.hess_);
		}

		POLYFEM_BOTH ScalarHess &operator+=(const ScalarHess &s)
		{
			*this = *this + s;
			return *this;
		}

		POLYFEM_BOTH ScalarHess &operator+=(const Scalar &s)
		{
			*this = *this + s;
			return *this;
		}

		// ======================================================================
		/// @{ \name Subtraction
		// ======================================================================

		POLYFEM_BOTH friend ScalarHess operator-(const ScalarHess &lhs,
												 const ScalarHess &rhs)
		{
			return ScalarHess(lhs.val_ - rhs.val_, lhs.grad_ - rhs.grad_,
							  lhs.hess_ - rhs.hess_);
		}

		POLYFEM_BOTH friend ScalarHess operator-(const ScalarHess &lhs,
												 const Scalar &rhs)
		{
			return ScalarHess(lhs.val_ - rhs, lhs.grad_, lhs.hess_);
		}

		POLYFEM_BOTH friend ScalarHess operator-(const Scalar &lhs,
												 const ScalarHess &rhs)
		{
			return ScalarHess(lhs - rhs.val_, -rhs.grad_, -rhs.hess_);
		}

		POLYFEM_BOTH friend ScalarHess operator-(const ScalarHess &s)
		{
			return ScalarHess(-s.val_, -s.grad_, -s.hess_);
		}

		POLYFEM_BOTH ScalarHess &operator-=(const ScalarHess &s)
		{
			*this = *this - s;
			return *this;
		}

		POLYFEM_BOTH ScalarHess &operator-=(const Scalar &s)
		{
			*this = *this - s;
			return *this;
		}

		// ======================================================================
		/// Division
		// ======================================================================

		POLYFEM_BOTH friend ScalarHess operator/(const ScalarHess &lhs,
												 const Scalar &rhs)
		{
			assert(rhs != 0);
			Scalar inv = Scalar{1} / rhs;
			return ScalarHess(lhs.val_ * inv, lhs.grad_ * inv, lhs.hess_ * inv);
		}

		POLYFEM_BOTH friend ScalarHess operator/(const Scalar &lhs,
												 const ScalarHess &rhs)
		{
			return lhs * inverse(rhs);
		}

		POLYFEM_BOTH friend ScalarHess operator/(const ScalarHess &lhs,
												 const ScalarHess &rhs)
		{
			return lhs * inverse(rhs);
		}

		POLYFEM_BOTH friend ScalarHess inverse(const ScalarHess &s)
		{
			assert(s.val_ != Scalar{0});
			Scalar val_Sqr = s.val_ * s.val_;
			Scalar val_Cub = val_Sqr * s.val_;
			Scalar invval_Sqr = Scalar{1} / val_Sqr;

			// vn = 1/v
			ScalarHess result(Scalar{1} / s.val_);

			// Dvn = -1/(v^2) Dv
			result.grad_ = s.grad_ * -invval_Sqr;

			// D^2vn = -1/(v^2) D^2v + 2/(v^3) Dv Dv^T
			result.hess_ = s.hess_ * -invval_Sqr;
			result.hess_ += s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() * (Scalar{2} / val_Cub);

			return result;
		}

		POLYFEM_BOTH ScalarHess &operator/=(const ScalarHess &s)
		{
			*this = *this / s;
			return *this;
		}

		POLYFEM_BOTH ScalarHess &operator/=(const Scalar &s)
		{
			*this = *this / s;
			return *this;
		}

		// ======================================================================
		/// Multiplication
		// ======================================================================

		POLYFEM_BOTH friend ScalarHess operator*(const ScalarHess &lhs,
												 const Scalar &rhs)
		{
			return ScalarHess(lhs.val_ * rhs, lhs.grad_ * rhs, lhs.hess_ * rhs);
		}

		POLYFEM_BOTH friend ScalarHess operator*(const Scalar &lhs,
												 const ScalarHess &rhs)
		{
			return ScalarHess(rhs.val_ * lhs, rhs.grad_ * lhs, rhs.hess_ * lhs);
		}

		POLYFEM_BOTH friend ScalarHess operator*(const ScalarHess &lhs,
												 const ScalarHess &rhs)
		{
			ScalarHess result(lhs.val_ * rhs.val_);

			/// Product rule
			result.grad_ = rhs.grad_ * lhs.val_ + lhs.grad_ * rhs.val_;

			// (i,j) = g*F_xixj + g*G_xixj + F_xi*G_xj + F_xj*G_xi
			result.hess_ = rhs.hess_ * lhs.val_;
			result.hess_ += lhs.hess_ * rhs.val_;
			// we only store the upper right block of full hessian.
			result.hess_ += lhs.grad_.template head<VALUE_DIM>() * rhs.grad_.template tail<VALUE_DIM>().transpose();
			result.hess_ += rhs.grad_.template head<VALUE_DIM>() * lhs.grad_.template tail<VALUE_DIM>().transpose();

			return result;
		}

		POLYFEM_BOTH ScalarHess &operator*=(const ScalarHess &s)
		{
			*this = *this * s;
			return *this;
		}

		POLYFEM_BOTH ScalarHess &operator*=(const Scalar &s)
		{
			*this = *this * s;
			return *this;
		}

		// ======================================================================
		/// Miscellaneous functions
		// ======================================================================

		POLYFEM_BOTH friend ScalarHess sqrt(const ScalarHess &s)
		{
			assert(s.val_ > Scalar{0});
			Scalar sqrtVal = math::sqrt(s.val_);
			Scalar temp = Scalar{1} / (Scalar{2} * sqrtVal);

			// vn = sqrt(v)
			ScalarHess result(sqrtVal);

			// Dvn = 1/(2 sqrt(v)) Dv
			result.grad_ = s.grad_ * temp;

			// D^2vn = 1/(2 sqrt(v)) D^2v - 1/(4 v*sqrt(v)) Dv Dv^T
			result.hess_ = s.hess_ * temp;
			result.hess_ += s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() * -Scalar{1} / (Scalar{4} * s.val_ * sqrtVal);

			return result;
		}

		POLYFEM_BOTH friend ScalarHess pow(const ScalarHess &s, const Scalar &a)
		{
			if (a == Scalar{0})
			{
				return ScalarHess(Scalar{1});
			}
			if (a == Scalar{1})
			{
				return s;
			}

			assert(s.val_ > Scalar{0});
			Scalar powVal = math::pow(s.val_, a);
			Scalar temp = a * math::pow(s.val_, a - 1);
			// vn = v ^ a
			ScalarHess result(powVal);

			// Dvn = a*v^(a-1) * Dv
			result.grad_ = s.grad_ * temp;

			// D^2vn = a*v^(a-1) D^2v - 1/(4 v*sqrt(v)) Dv Dv^T
			result.hess_ = s.hess_ * temp;
			result.hess_ += s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() * (a * (a - 1) * math::pow(s.val_, a - 2));

			return result;
		}

		POLYFEM_BOTH friend ScalarHess exp(const ScalarHess &s)
		{
			Scalar expVal = math::exp(s.val_);

			// vn = exp(v)
			ScalarHess result(expVal);

			// Dvn = exp(v) * Dv
			result.grad_ = s.grad_ * expVal;

			// D^2vn = exp(v) * Dv*Dv^T + exp(v) * D^2v
			result.hess_ = (s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() + s.hess_) * expVal;

			return result;
		}

		POLYFEM_BOTH friend ScalarHess log(const ScalarHess &s)
		{
			assert(s.val_ > Scalar{0});
			Scalar logVal = math::log(s.val_);

			// vn = log(v)
			ScalarHess result(logVal);

			// Dvn = Dv / v
			result.grad_ = s.grad_ / s.val_;

			// D^2vn = (v*D^2v - Dv*Dv^T)/(v^2)
			result.hess_ =
				s.hess_ / s.val_ - (s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() / (s.val_ * s.val_));

			return result;
		}

		POLYFEM_BOTH friend ScalarHess sin(const ScalarHess &s)
		{
			Scalar sinVal = math::sin(s.val_);
			Scalar cosVal = math::cos(s.val_);

			// vn = sin(v)
			ScalarHess result(sinVal);

			// Dvn = cos(v) * Dv
			result.grad_ = s.grad_ * cosVal;

			// D^2vn = -sin(v) * Dv*Dv^T + cos(v) * Dv^2
			result.hess_ = s.hess_ * cosVal;
			result.hess_ += s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() * -sinVal;

			return result;
		}

		POLYFEM_BOTH friend ScalarHess cos(const ScalarHess &s)
		{
			Scalar sinVal = math::sin(s.val_);
			Scalar cosVal = math::cos(s.val_);
			// vn = cos(v)
			ScalarHess result(cosVal);

			// Dvn = -sin(v) * Dv
			result.grad_ = s.grad_ * -sinVal;

			// D^2vn = -cos(v) * Dv*Dv^T - sin(v) * Dv^2
			result.hess_ = s.hess_ * -sinVal;
			result.hess_ += s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() * -cosVal;

			return result;
		}

		POLYFEM_BOTH friend ScalarHess acos(const ScalarHess &s)
		{
			assert(math::abs(s.val_) < 1);

			Scalar temp = -math::sqrt(Scalar{1} - s.val_ * s.val_);

			// vn = acos(v)
			ScalarHess result(math::acos(s.val_));

			// Dvn = -1/sqrt(1-v^2) * Dv
			result.grad_ = s.grad_ * (Scalar{1} / temp);

			// D^2vn = -1/sqrt(1-v^2) * D^2v - v/[(1-v^2)^(3/2)] * Dv*Dv^T
			result.hess_ = s.hess_ * (Scalar{1} / temp);
			result.hess_ += s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() * s.val_ / (temp * temp * temp);

			return result;
		}

		POLYFEM_BOTH friend ScalarHess asin(const ScalarHess &s)
		{
			assert(math::abs(s.val_) < 1);

			Scalar temp = math::sqrt((Scalar)1 - s.val_ * s.val_);

			// vn = asin(v)
			ScalarHess result(math::asin(s.val_));

			// Dvn = 1/sqrt(1-v^2) * Dv
			result.grad_ = s.grad_ * (Scalar{1} / temp);

			// D^2vn = 1/sqrt(1-v*v) * D^2v + v/[(1-v^2)^(3/2)] * Dv*Dv^T
			result.hess_ = s.hess_ * (Scalar{1} / temp);
			result.hess_ += s.grad_.template head<VALUE_DIM>() * s.grad_.template tail<VALUE_DIM>().transpose() * s.val_ / (temp * temp * temp);

			return result;
		}

		POLYFEM_BOTH friend ScalarHess atan2(const ScalarHess &y,
											 const ScalarHess &x)
		{
			// vn = atan2(y, x)
			ScalarHess result(math::atan2(y.val_, x.val_));

			// Dvn = (x*Dy - y*Dx) / (x^2 + y^2)
			Scalar denom = x.val_ * x.val_ + y.val_ * y.val_;
			assert(denom != Scalar{0});
			Scalar denomSqr = denom * denom;
			result.grad_ = y.grad_ * (x.val_ / denom) - x.grad_ * (y.val_ / denom);

			// D^2vn = (Dy*Dx^T + xD^2y - Dx*Dy^T - yD^2x) / (x^2+y^2)
			//    - [(x*Dy - y*Dx) * (2*x*Dx + 2*y*Dy)^T] / (x^2+y^2)^2
			result.hess_ = (y.hess_ * x.val_ + y.grad_.template head<VALUE_DIM>() * x.grad_.template tail<VALUE_DIM>().transpose() - x.hess_ * y.val_ - x.grad_.template head<VALUE_DIM>() * y.grad_.template tail<VALUE_DIM>().transpose()) / denom;

			result.hess_ -= (y.grad_.template head<VALUE_DIM>() * (x.val_ / denomSqr) - x.grad_.template head<VALUE_DIM>() * (y.val_ / denomSqr)) * (x.grad_.template tail<VALUE_DIM>() * Scalar{2} * x.val_ + y.grad_.template tail<VALUE_DIM>() * Scalar{2} * y.val_).transpose();

			return result;
		}

		// ======================================================================
		/// Comparison and assignment
		// ======================================================================

		POLYFEM_BOTH friend bool operator<(const ScalarHess &lhs, const Scalar &rhs)
		{
			return lhs.val_ < rhs;
		}

		POLYFEM_BOTH friend bool operator<(const Scalar &lhs, const ScalarHess &rhs)
		{
			return lhs < rhs.val_;
		}

		POLYFEM_BOTH friend bool operator<(const ScalarHess &lhs,
										   const ScalarHess &rhs)
		{
			return lhs.val_ < rhs.val_;
		}

		POLYFEM_BOTH friend bool operator<=(const ScalarHess &lhs,
											const Scalar &rhs)
		{
			return lhs.val_ <= rhs;
		}

		POLYFEM_BOTH friend bool operator<=(const Scalar &lhs,
											const ScalarHess &rhs)
		{
			return lhs <= rhs.val_;
		}

		POLYFEM_BOTH friend bool operator<=(const ScalarHess &lhs,
											const ScalarHess &rhs)
		{
			return lhs.val_ <= rhs.val_;
		}

		POLYFEM_BOTH friend bool operator>(const ScalarHess &lhs, const Scalar &rhs)
		{
			return lhs.val_ > rhs;
		}

		POLYFEM_BOTH friend bool operator>(const Scalar &lhs, const ScalarHess &rhs)
		{
			return lhs > rhs.val_;
		}

		POLYFEM_BOTH friend bool operator>(const ScalarHess &lhs,
										   const ScalarHess &rhs)
		{
			return lhs.val_ > rhs.val_;
		}

		POLYFEM_BOTH friend bool operator>=(const ScalarHess &lhs,
											const Scalar &rhs)
		{
			return lhs.val_ >= rhs;
		}

		POLYFEM_BOTH friend bool operator>=(const Scalar &lhs,
											const ScalarHess &rhs)
		{
			return lhs >= rhs.val_;
		}

		POLYFEM_BOTH friend bool operator>=(const ScalarHess &lhs,
											const ScalarHess &rhs)
		{
			return lhs.val_ >= rhs.val_;
		}

		POLYFEM_BOTH friend bool operator==(const ScalarHess &lhs,
											const Scalar &rhs)
		{
			return lhs.val_ == rhs;
		}

		POLYFEM_BOTH friend bool operator==(const Scalar &lhs,
											const ScalarHess &rhs)
		{
			return lhs == rhs.val_;
		}

		POLYFEM_BOTH friend bool operator==(const ScalarHess &lhs,
											const ScalarHess &rhs)
		{
			return lhs.val_ == rhs.val_;
		}

		POLYFEM_BOTH friend bool operator!=(const ScalarHess &lhs,
											const Scalar &rhs)
		{
			return lhs.val_ != rhs;
		}

		POLYFEM_BOTH friend bool operator!=(const Scalar &lhs,
											const ScalarHess &rhs)
		{
			return lhs != rhs.val_;
		}

		POLYFEM_BOTH friend bool operator!=(const ScalarHess &lhs,
											const ScalarHess &rhs)
		{
			return lhs.val_ != rhs.val_;
		}

		POLYFEM_BOTH friend ScalarHess abs(const ScalarHess &s)
		{
			return s.val_ < Scalar{0} ? -s : s;
		}

		POLYFEM_BOTH friend ScalarHess min(const ScalarHess &lhs,
										   const ScalarHess &rhs)
		{
			return rhs.val_ < lhs.val_ ? rhs : lhs;
		}

		POLYFEM_BOTH friend ScalarHess min(const ScalarHess &lhs, const Scalar &rhs)
		{
			return rhs < lhs.val_ ? ScalarHess(rhs) : lhs;
		}

		POLYFEM_BOTH friend ScalarHess min(const Scalar &lhs, const ScalarHess &rhs)
		{
			return rhs.val_ < lhs ? rhs : ScalarHess(lhs);
		}

		POLYFEM_BOTH friend ScalarHess max(const ScalarHess &lhs,
										   const ScalarHess &rhs)
		{
			return lhs.val_ < rhs.val_ ? rhs : lhs;
		}

		POLYFEM_BOTH friend ScalarHess max(const ScalarHess &lhs, const Scalar &rhs)
		{
			return lhs.val_ < rhs ? ScalarHess(rhs) : lhs;
		}

		POLYFEM_BOTH friend ScalarHess max(const Scalar &lhs, const ScalarHess &rhs)
		{
			return lhs < rhs.val_ ? rhs : ScalarHess(lhs);
		}
	};

	template <int value_dim>
	using Double1 = ScalarGrad<double, value_dim>;
	template <int value_dim>
	using Double2 = ScalarHess<double, value_dim>;

} // namespace polyfem::autodiff

// Overload AD type Eigen NumTraits else stuff like abs wont work properly.
namespace Eigen
{

	template <int value_dim>
	struct NumTraits<polyfem::autodiff::ScalarGrad<double, value_dim>>
		: GenericNumTraits<polyfem::autodiff::ScalarGrad<double, value_dim>>
	{
		using Real = polyfem::autodiff::ScalarGrad<double, value_dim>;
		using NonInteger = Real;
		using Nested = Real;
		using Literal = double;

		enum
		{
			IsInteger = 0,
			IsSigned = NumTraits<double>::IsSigned,
			IsComplex = 0,
			RequireInitialization = 1,
			ReadCost = NumTraits<double>::ReadCost + value_dim,
			AddCost = NumTraits<double>::AddCost * (1 + value_dim),
			MulCost = NumTraits<double>::MulCost * (1 + value_dim) + NumTraits<double>::AddCost * value_dim
		};

		EIGEN_DEVICE_FUNC static Real epsilon()
		{
			return Real(NumTraits<double>::epsilon());
		}

		EIGEN_DEVICE_FUNC static Real dummy_precision()
		{
			return Real(NumTraits<double>::dummy_precision());
		}
	};

	template <int value_dim>
	struct NumTraits<polyfem::autodiff::ScalarHess<double, value_dim>>
		: GenericNumTraits<polyfem::autodiff::ScalarHess<double, value_dim>>
	{
		using Real = polyfem::autodiff::ScalarHess<double, value_dim>;
		using NonInteger = Real;
		using Nested = Real;
		using Literal = double;

		enum
		{
			IsInteger = 0,
			IsSigned = NumTraits<double>::IsSigned,
			IsComplex = 0,
			RequireInitialization = 1,
			ReadCost =
				NumTraits<double>::ReadCost + 2 * value_dim + value_dim * value_dim,
			AddCost = NumTraits<double>::AddCost * (1 + 2 * value_dim + value_dim * value_dim),
			MulCost =
				NumTraits<double>::MulCost * (1 + 2 * value_dim + value_dim * value_dim) + NumTraits<double>::AddCost * (2 * value_dim + value_dim * value_dim)
		};

		EIGEN_DEVICE_FUNC static Real epsilon()
		{
			return Real(NumTraits<double>::epsilon());
		}

		EIGEN_DEVICE_FUNC static Real dummy_precision()
		{
			return Real(NumTraits<double>::dummy_precision());
		}
	};

} // namespace Eigen
