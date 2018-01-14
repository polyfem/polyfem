#include "ElementBases.hpp"

namespace poly_fem
{
	bool ElementBases::is_complete() const
	{
		for(auto &b : bases)
		{
			if(!b.is_complete())
				return false;
		}

		return true;
	}
	void ElementBases::eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const
	{
		if(!has_parameterization)
		{
			mapped = samples;
			return;
		}

		mapped = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());
		Eigen::MatrixXd tmp;
		evaluate_bases(samples, tmp);

		const int n_local_bases = int(bases.size());
		for(int j = 0; j < n_local_bases; ++j)
		{
			const Basis &b = bases[j];

			for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			{
				for (long k = 0; k < tmp.rows(); ++k){
					mapped.row(k) += tmp(k,j) * b.global()[ii].node * b.global()[ii].val;
				}
			}
		}
	}

	void ElementBases::evaluate_bases(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		val.resize(uv.rows(), bases.size());
		Eigen::MatrixXd tmp;
		for(size_t i = 0; i < bases.size(); ++i){
			bases[i].eval_basis(uv, tmp);
			val.col(i) = tmp;
		}
	}

	void ElementBases::evaluate_grads(const Eigen::MatrixXd &uv, const int dim, Eigen::MatrixXd &grad) const
	{
		grad.resize(uv.rows(), bases.size());
		Eigen::MatrixXd grad_tmp;
		for(size_t i = 0; i < bases.size(); ++i)
		{
			bases[i].eval_grad(uv, grad_tmp);
			grad.col(i) = grad_tmp.col(dim);
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

		Eigen::MatrixXd local_gradx, local_grady, local_gradz;
		evaluate_grads(samples, 0, local_gradx);
		evaluate_grads(samples, 1, local_grady);

		if(is_volume)
			evaluate_grads(samples, 2, local_gradz);

		for(int j = 0; j < n_local_bases; ++j)
		{
			const Basis &b = bases[j];

			for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			{
				for(long k = 0; k < samples.rows(); ++k)
				{
					dxmv.row(k) += local_gradx(k,j) * b.global()[ii].node  * b.global()[ii].val;
					dymv.row(k) += local_grady(k,j) * b.global()[ii].node  * b.global()[ii].val;
					if(is_volume)
						dzmv.row(k) += local_gradz(k,j) * b.global()[ii].node  * b.global()[ii].val;
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

