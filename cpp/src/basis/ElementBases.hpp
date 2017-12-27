#ifndef ELEMENT_BASES_HPP
#define ELEMENT_BASES_HPP

#include "Basis.hpp"
#include "Quadrature.hpp"

#include <vector>

namespace poly_fem
{
	class ElementBases
	{
	public:
		std::vector<Basis> bases;

		Quadrature quadrature;

		bool has_parameterization = true;

		void eval_geom_mapping_grads(const Eigen::MatrixXd &samples, std::vector<Eigen::MatrixXd> &grads) const;
		void eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const;
	};
}

#endif
