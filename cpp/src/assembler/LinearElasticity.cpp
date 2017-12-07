#include "LinearElasticity.hpp"

namespace poly_fem
{
	void LinearElasticity::assemble(const Eigen::MatrixXd &gradi, const Eigen::MatrixXd &gradj, const Eigen::MatrixXd &da, Eigen::MatrixXd &res) const
	{
		// mu ((gradi' gradj) Id + gradi gradj') + lambda gradi *gradj';
		res.resize(gradi.rows(), size()*size());
		const auto &dot = (gradi.array() * gradj.array()).rowwise().sum();

		for(long k = 0; k < gradi.rows(); ++k)
		{
			const Eigen::MatrixXd outer = gradi.row(k).transpose() * gradj.row(k);
			for(int i = 0; i < size(); ++i)
			{
				for(int j = 0; j < size(); ++j)
				{
					res(k, i * size() + j) = outer(i * size() + j)*(lambda_ + mu_);
					if(i == j) res(k, i * size() + j) += mu_ * dot(k);

					res(k, i * size() + j) *= da(k);
				}
			}
		}
	}

}