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

		void eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const
		{
			if(!has_parameterization)
			{
				mapped = samples;
				return;
			}

			mapped = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());
			Eigen::MatrixXd tmp;

			const int n_local_bases = int(bases.size());
			for(int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b = bases[j];

				b.basis(samples, tmp);

				for(std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					for (long k = 0; k < tmp.rows(); ++k){
						mapped.row(k) += tmp(k,0) * b.global()[ii].node * b.global()[ii].val;
					}
				}
			}
		}
	};
}

#endif
