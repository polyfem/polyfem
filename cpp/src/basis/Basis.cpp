#include "Basis.hpp"

#include <iostream>

namespace poly_fem
{
	Basis::Basis()
	{ }


	void Basis::init(const int global_index, const int local_index, const Eigen::MatrixXd &node)
	{
		global_index_ = global_index;
		local_index_ = local_index;
		node_ = node;
	}

	void Basis::basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		basis_(uv, val);
	}

	void Basis::grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		grad_(uv, val);
	}

	void Basis::eval_geom_mapping(const bool has_parameterization, const Eigen::MatrixXd &samples, const std::vector<Basis> &local_bases, Eigen::MatrixXd &mapped)
	{
		if(!has_parameterization)
		{
			mapped = samples;
			return;
		}
		
		mapped = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());
		Eigen::MatrixXd tmp;

		const int n_local_bases = int(local_bases.size());
		for(int j = 0; j < n_local_bases; ++j)
		{
			const Basis &b = local_bases[j];

			b.basis(samples, tmp);

			for (long k = 0; k < tmp.rows(); ++k){
				mapped.row(k) += tmp(k,0) * b.node();
			}
		}
	}
}