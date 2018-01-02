#include "ElementBases.hpp"

namespace poly_fem
{
	void ElementBases::eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const
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

	void ElementBases::eval_geom_mapping_grads(const Eigen::MatrixXd &samples, std::vector<Eigen::MatrixXd> &grads) const
	{
		grads.resize(samples.rows());

		if(!has_parameterization)
		{
			std::fill(grads.begin(), grads.end(), Eigen::MatrixXd::Identity(samples.cols(), samples.cols()));
			return;
		}

		Eigen::MatrixXd local_grad;

		const int n_local_bases = int(bases.size());
		const bool is_volume = samples.cols() == 3;
		Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());
		Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());
		Eigen::MatrixXd dzmv = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());

		for(int j = 0; j < n_local_bases; ++j)
		{
			const Basis &b = bases[j];

			b.grad(samples, local_grad);

			for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			{
				for(long k = 0; k < samples.rows(); ++k)
				{
					dxmv.row(k) += local_grad(k,0) * b.global()[ii].node  * b.global()[ii].val;
					dymv.row(k) += local_grad(k,1) * b.global()[ii].node  * b.global()[ii].val;
					if(is_volume)
						dzmv.row(k) += local_grad(k,2) * b.global()[ii].node  * b.global()[ii].val;
				}
			}
		}

		Eigen::MatrixXd tmp(samples.cols(), samples.cols());

		for(long k = 0; k < samples.rows(); ++k)
		{
			tmp.row(0) = dxmv.row(k);
			tmp.row(1) = dymv.row(k);
			if(is_volume)
				tmp.row(2) = dzmv.row(k);

			grads[k] = tmp;
		}
	}
}

