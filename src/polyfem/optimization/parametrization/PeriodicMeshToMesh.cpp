#include "PeriodicMeshToMesh.hpp"
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	PeriodicMeshToMesh::PeriodicMeshToMesh(const Eigen::MatrixXd &V)
	{
		dim_ = V.cols();

		assert(dim_ == V.cols());
		const int n_verts = V.rows();

		Eigen::VectorXd min = V.colwise().minCoeff();
		Eigen::VectorXd max = V.colwise().maxCoeff();
		Eigen::VectorXd scale_ = max - min;

		n_periodic_dof_ = 0;
		dependent_map.resize(n_verts);
		dependent_map.setConstant(-1);

		const double eps = 1e-4 * scale_.maxCoeff();
		Eigen::VectorXi boundary_indices;
		{
			Eigen::VectorXi boundary_mask1 = ((V.rowwise() - min.transpose()).rowwise().minCoeff().array() < eps).select(Eigen::VectorXi::Ones(V.rows()), Eigen::VectorXi::Zero(V.rows()));
			Eigen::VectorXi boundary_mask2 = ((V.rowwise() - max.transpose()).rowwise().maxCoeff().array() > -eps).select(Eigen::VectorXi::Ones(V.rows()), Eigen::VectorXi::Zero(V.rows()));
			Eigen::VectorXi boundary_mask = boundary_mask1.array() + boundary_mask2.array();

			boundary_indices.setZero(boundary_mask.sum());
			for (int i = 0, j = 0; i < boundary_mask.size(); i++)
				if (boundary_mask[i])
					boundary_indices[j++] = i;
		}

		// find corresponding periodic boundary nodes
		Eigen::MatrixXd V_boundary = V(boundary_indices, Eigen::all);
		for (int d = 0; d < dim_; d++)
		{
			Eigen::VectorXi mask1 = (V_boundary.col(d).array() < min(d) + eps).select(Eigen::VectorXi::Ones(V_boundary.rows()), Eigen::VectorXi::Zero(V_boundary.rows()));
			Eigen::VectorXi mask2 = (V_boundary.col(d).array() > max(d) - eps).select(Eigen::VectorXi::Ones(V_boundary.rows()), Eigen::VectorXi::Zero(V_boundary.rows()));

			for (int i = 0; i < mask1.size(); i++)
			{
				if (!mask1(i))
					continue;

				bool found_target = false;
				for (int j = 0; j < mask2.size(); j++)
				{
					if (!mask2(j))
						continue;

					RowVectorNd projected_diff = V_boundary.row(j) - V_boundary.row(i);
					projected_diff(d) = 0;
					if (projected_diff.norm() < eps)
					{
						dependent_map(boundary_indices[j]) = boundary_indices[i];
						std::array<int, 2> pair = {{boundary_indices[i], boundary_indices[j]}};
						periodic_dependence[d].insert(pair);
						found_target = true;
						break;
					}
				}
				if (!found_target)
					log_and_throw_error("Periodic mesh failed to find corresponding node for {} in {} direction!", V_boundary.row(i), (std::vector<char>{'X', 'Y', 'Z'})[d]);
			}
		}

		// break dependency chains into direct dependency
		for (int d = 0; d < dim_; d++)
			for (int i = 0; i < dependent_map.size(); i++)
				if (dependent_map(i) >= 0 && dependent_map(dependent_map(i)) >= 0)
					dependent_map(i) = dependent_map(dependent_map(i));

		Eigen::VectorXi reduce_map;
		reduce_map.setZero(dependent_map.size());
		for (int i = 0; i < dependent_map.size(); i++)
			if (dependent_map(i) < 0)
				reduce_map(i) = n_periodic_dof_++;
		for (int i = 0; i < dependent_map.size(); i++)
			if (dependent_map(i) >= 0)
				reduce_map(i) = reduce_map(dependent_map(i));

		dependent_map = std::move(reduce_map);
	}

	Eigen::VectorXd PeriodicMeshToMesh::eval(const Eigen::VectorXd &x) const
	{
		assert(x.size() == input_size());

		Eigen::MatrixXd affine = utils::unflatten(x.tail(dim_ * dim_), dim_).transpose();
		Eigen::VectorXd y;
		y.setZero(size(x.size()));
		for (int i = 0; i < dependent_map.size(); i++)
			y.segment(i * dim_, dim_) = affine * x.segment(dependent_map(i) * dim_, dim_);

		for (int d = 0; d < dim_; d++)
		{
			const auto &dependence_list = periodic_dependence[d];
			for (const auto &pair : dependence_list)
				y.segment(pair[1] * dim_, dim_) += affine.col(d);
		}

		return y;
	}

	Eigen::VectorXd PeriodicMeshToMesh::inverse_eval(const Eigen::VectorXd &y)
	{
		assert(y.size() == dim_ * dependent_map.size());
		Eigen::VectorXd x;
		x.setZero(input_size());

		Eigen::MatrixXd V = utils::unflatten(y, dim_);
		Eigen::VectorXd min = V.colwise().minCoeff();
		Eigen::VectorXd max = V.colwise().maxCoeff();
		Eigen::VectorXd scale = max - min;
		Eigen::MatrixXd affine = scale.asDiagonal();
		x.tail(dim_ * dim_) = Eigen::Map<Eigen::VectorXd>(affine.data(), dim_ * dim_, 1);

		Eigen::VectorXd z = y;
		for (int d = 0; d < dim_; d++)
		{
			const auto &dependence_list = periodic_dependence[d];
			for (const auto &pair : dependence_list)
				z(pair[1] * dim_ + d) -= scale[d];
		}

		for (int i = 0; i < dependent_map.size(); i++)
			x.segment(dependent_map(i) * dim_, dim_) = z.segment(i * dim_, dim_).array() / scale.array();

		if ((y - eval(x)).norm() > 1e-5)
			log_and_throw_adjoint_error("Non-periodic mesh detected!");

		return x;
	}

	Eigen::VectorXd PeriodicMeshToMesh::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		assert(x.size() == input_size());
		Eigen::VectorXd reduced_grad;
		reduced_grad.setZero(x.size());

		Eigen::MatrixXd affine = utils::unflatten(x.tail(dim_ * dim_), dim_).transpose();

		for (int i = 0; i < dependent_map.size(); i++)
			reduced_grad.segment(dependent_map(i) * dim_, dim_) += affine.transpose() * grad.segment(i * dim_, dim_);

		for (int i = 0; i < dependent_map.size(); i++)
		{
			Eigen::MatrixXd tmp = grad.segment(i * dim_, dim_) * x.segment(dependent_map(i) * dim_, dim_).transpose();
			reduced_grad.tail(dim_ * dim_) += utils::flatten(tmp.transpose());
		}

		for (int d = 0; d < dim_; d++)
		{
			const auto &dependence_list = periodic_dependence[d];
			for (const auto &pair : dependence_list)
				reduced_grad.segment(dim_ * n_periodic_dof_ + d * dim_, dim_) += grad.segment(pair[1] * dim_, dim_);
		}

		return reduced_grad;
	}
} // namespace polyfem::solver