#ifndef BASIS_HPP
#define BASIS_HPP

#include "Quadrature.hpp"

#include <Eigen/Dense>
#include <functional>

#include <vector>

namespace poly_fem
{
	class Basis
	{

	public:
		typedef std::function<void(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)> Fun;


		Basis();

		void init(const int global_index, const int local_index, const Eigen::MatrixXd &node);


		void basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;
		void grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;

		inline int global_index() const { return global_index_; }
		inline const Eigen::MatrixXd &node() const { return node_; }
		inline void set_node(const Eigen::MatrixXd &v) { node_ = v; }

		inline void set_basis(const Fun &fun) { basis_ = fun; }
		inline void set_grad(const Fun &fun) { grad_ = fun; }


		static void eval_geom_mapping(const bool has_parameterization, const Eigen::MatrixXd &samples, const std::vector<Basis> &local_bases, Eigen::MatrixXd &mapped);
	private:
		int global_index_;
		int local_index_;

		Eigen::MatrixXd node_;


		Fun basis_;
		Fun grad_;
	};
}

#endif //BASIS_HPP
