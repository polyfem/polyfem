#pragma once

#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <functional>

#include <vector>

namespace polyfem
{
	namespace basis
	{
		///
		/// @brief      Represents a virtual node of the FEM mesh as a weighted sum
		///             of real (unknown) nodes. This class stores the id, weights
		///             and positions of the real mesh nodes to use in the weighted
		///             sum.
		///
		class Local2Global
		{
		public:
			int index;  ///< global index of the actual node
			double val; ///< weight

			RowVectorNd node; ///< node position

			Local2Global()
				: index(-1), val(0)
			{
			}

			Local2Global(const int _index, const RowVectorNd &_node, const double _val)
				: index(_index), val(_val), node(_node)
			{
			}
		};

		///
		/// @brief      Represents one basis function and its gradient.
		///
		class Basis
		{

		public:
			typedef std::function<void(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)> Fun;

			Basis();

			///
			/// @brief      Initialize a basis function within an element
			///
			/// @param[in]  global_index   Global index of the node associated to the basis
			/// @param[in]  local_index    Local index of the node within the element
			/// @param[in]  node           1 x dim position of the node associated to the basis
			///
			void init(const int order, const int global_index, const int local_index, const RowVectorNd &node);

			///
			/// @brief      Checks if global is empty or not
			///
			inline bool is_complete() const { return !global_.empty(); }

			///
			/// @brief      Evaluates the basis function over a set of uv
			///             parameters.
			///
			/// @param[in]  uv     #uv x dim matrix of parameters to evaluate
			/// @param[out] val    #uv x 1 vector of computed values
			///
			void eval_basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
			{
				assert(basis_);
				basis_(uv, val);
			}

			///
			/// @brief Evaluate the basis function over a set of uv parameters.
			///
			/// @param[in] uv #uv x dim matrix of parameters to evaluate
			/// @return #uv x 1 vector of computed values
			///
			Eigen::MatrixXd operator()(const Eigen::MatrixXd &uv) const
			{
				Eigen::MatrixXd val;
				eval_basis(uv, val);
				return val;
			}

			///
			/// @brief      Evaluate the gradient of the basis function.
			///
			/// @param[in]  uv     #uv x dim matrix of parameters to evaluate
			/// @param[out] val    #uv x dim matrix of computed gradients
			///
			void eval_grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
			{
				assert(grad_);
				grad_(uv, val);
			}

			// list of local to global mappings
			inline const std::vector<Local2Global> &global() const { return global_; }
			inline std::vector<Local2Global> &global() { return global_; }

			// setting the basis lambda and its gradient
			inline void set_basis(const Fun &fun) { basis_ = fun; }
			inline void set_grad(const Fun &fun) { grad_ = fun; }

			inline bool is_defined() const { return (basis_ ? true : false); }
			inline int order() const { return order_; }

			// output
			friend std::ostream &operator<<(std::ostream &os, const Basis &obj)
			{
				os << obj.local_index_ << ":\n";
				for (auto l2g : obj.global_)
					os << "\tl2g: " << l2g.index << " (" << l2g.node << ") " << l2g.val << "\n";

				return os;
			}

		private:
			std::vector<Local2Global> global_; ///< list of real nodes influencing the basis
			int local_index_;                  ///< local index inside the element (for debugging purposes)
			int order_;

			Fun basis_; ///< basis and gadient
			Fun grad_;
		};
	} // namespace basis
} // namespace polyfem
