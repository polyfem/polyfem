#pragma once

#include <Eigen/Core>

#include <cassert>
#include <utility>

namespace polyfem::assembler
{
	/// Reusable element matrix with inline storage for small blocks and a dynamic
	/// fallback for high-order elements. Visiting exposes the actual Eigen type.
	class LocalHessian
	{
	public:
		// Covers quadratic tetrahedra without putting high-order blocks on the stack.
		static constexpr int SmallDofs = 30;
		using SmallMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
										  Eigen::ColMajor, SmallDofs, SmallDofs>;

		void resize(const Eigen::Index rows, const Eigen::Index cols)
		{
			assert(rows >= 0 && cols >= 0);
			use_small_ = rows <= SmallDofs && cols <= SmallDofs;
			if (use_small_)
				small_.resize(rows, cols);
			else
				large_.resize(rows, cols);
		}

		Eigen::Index rows() const { return use_small_ ? small_.rows() : large_.rows(); }
		Eigen::Index cols() const { return use_small_ ? small_.cols() : large_.cols(); }

		template <typename Derived>
		void assign(const Eigen::MatrixBase<Derived> &matrix)
		{
			resize(matrix.rows(), matrix.cols());
			visit([&](auto &H) { H = matrix; });
		}

		template <typename F>
		decltype(auto) visit(F &&f)
		{
			if (use_small_)
				return std::forward<F>(f)(small_);
			return std::forward<F>(f)(large_);
		}

		template <typename F>
		decltype(auto) visit(F &&f) const
		{
			if (use_small_)
				return std::forward<F>(f)(small_);
			return std::forward<F>(f)(large_);
		}

	private:
		SmallMatrix small_;
		// Keep this buffer when selecting small_ so mixed-order meshes can reuse it.
		Eigen::MatrixXd large_;
		bool use_small_ = true;
	};
} // namespace polyfem::assembler
