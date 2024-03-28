#pragma once

#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <functional>
#include <vector>

namespace polyfem::basis
{
	/// @brief A virtual node of the FEM mesh as a weighted sum of real (unknown) nodes.
	/// This class stores the id, weights and positions of the real mesh nodes
	/// to use in the weighted sum.
	struct Local2Global
	{
		int index;        ///< @brief global index of the actual node
		double val;       ///< @brief weight
		RowVectorNd node; ///< @brief node position

		Local2Global()
			: index(-1), val(0)
		{
		}

		Local2Global(const int _index, const RowVectorNd &_node, const double _val)
			: index(_index), val(_val), node(_node)
		{
		}
	};

	/// @brief One basis function and its gradient.
	class Basis
	{
	public:
		typedef std::function<void(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)> Fun;

		/// @brief Default constructor
		Basis();

		/// @brief Initialize a basis function within an element
		/// @param[in] global_index Global index of the node associated to the basis
		/// @param[in] local_index Local index of the node within the element
		/// @param[in] node 1 x dim position of the node associated to the basis
		void init(const int order, const int global_index, const int local_index, const RowVectorNd &node);

		/// @brief Checks if global is empty or not
		bool is_complete() const { return !global_.empty(); }

		/// @brief Evaluates the basis function over a set of uv parameters.
		/// @param[in] uv #uv x dim matrix of parameters to evaluate
		/// @param[out] val #uv x 1 vector of computed values
		void eval_basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
		{
			assert(basis_);
			basis_(uv, val);
		}

		/// @brief Evaluate the basis function over a set of uv parameters.
		/// @param[in] uv #uv x dim matrix of parameters to evaluate
		/// @return #uv x 1 vector of computed values
		Eigen::MatrixXd operator()(const Eigen::MatrixXd &uv) const
		{
			Eigen::MatrixXd val;
			eval_basis(uv, val);
			return val;
		}

		/// @brief Evaluate the gradient of the basis function.
		/// @param[in] uv #uv x dim matrix of parameters to evaluate
		/// @param[out] val #uv x dim matrix of computed gradients
		void eval_grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
		{
			assert(grad_);
			grad_(uv, val);
		}

		// list of local to global mappings
		const std::vector<Local2Global> &global() const { return global_; }
		std::vector<Local2Global> &global() { return global_; }

		// setting the basis lambda and its gradient
		void set_basis(const Fun &fun) { basis_ = fun; }
		void set_grad(const Fun &fun) { grad_ = fun; }

		bool is_defined() const { return bool(basis_); }
		int order() const { return order_; }

		// output
		friend std::ostream &operator<<(std::ostream &os, const Basis &obj);

	private:
		std::vector<Local2Global> global_; ///< list of real nodes influencing the basis
		int local_index_;                  ///< local index inside the element
		int order_;                        ///< polynomial order of the basis function

		Fun basis_; ///< basis function
		Fun grad_;  ///< gradient of the basis
	};
} // namespace polyfem::basis
