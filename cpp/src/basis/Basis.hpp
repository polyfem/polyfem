#ifndef BASIS_HPP
#define BASIS_HPP

#include "Quadrature.hpp"

#include <Eigen/Dense>
#include <functional>

#include <vector>

namespace poly_fem
{
	class Local2Global
	{
	public:
		int index; // global index of the actual dof
		double val; // weight

		Eigen::MatrixXd node; // dof position
	};

	///
	/// @brief      Represents one basis function and its gradient.
	///
	class Basis
	{

	public:
		typedef std::function<void(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)> Fun;


		Basis();

		void init(const int global_index, const int local_index, const Eigen::MatrixXd &node);

		///
		/// @brief      Evaluates the basis function over a set of uv
		///             parameters.
		///
		/// @param[in]  uv    { #uv x dim matrix of parameters to evaluate }
		/// @param[out] val   { #uv x 1 vector of computed values }
		///
		void basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const { basis_(uv, val); }

		///
		/// @brief      Evaluate the gradient of the basis function.
		///
		/// @param[in]  uv    { #uv x dim matrix of parameters to evaluate }
		/// @param[out] val   { #uv x dim matrix of computed gradients }
		///
		void grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const { grad_(uv, val); }

		inline const std::vector< Local2Global > &global() const { return global_; }
		inline std::vector< Local2Global > &global() { return global_; }
		// inline const Eigen::MatrixXd &node() const { return node_; }
		// inline void set_node(const Eigen::MatrixXd &v) { node_ = v; }

		inline void set_basis(const Fun &fun) { basis_ = fun; }
		inline void set_grad(const Fun &fun) { grad_ = fun; }
	private:
		std::vector< Local2Global > global_; // real global dofs influencing the basis
		int local_index_; // local index inside the element (for debugging purposes)


		Fun basis_;
		Fun grad_;
	};
}

#endif //BASIS_HPP
