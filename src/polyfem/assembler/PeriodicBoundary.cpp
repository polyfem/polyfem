#include "PeriodicBoundary.hpp"
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/Logger.hpp>

namespace
{
	Eigen::MatrixXd extract_vertices(const std::shared_ptr<polyfem::mesh::MeshNodes> &mesh_nodes)
	{
		Eigen::MatrixXd V;
		V.setZero(mesh_nodes->n_nodes(), mesh_nodes->node_position(0).size());

		for (int i = 0; i < V.rows(); i++)
			V.row(i) = mesh_nodes->node_position(i);

		return V;
	}
} // namespace

namespace polyfem::utils
{
	PeriodicBoundary::PeriodicBoundary(
		const bool is_scalar,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::shared_ptr<mesh::MeshNodes> &mesh_nodes,
		const Eigen::MatrixXd &affine_matrix,
		const double tol) : affine_matrix_(affine_matrix)
	{
		const int dim = affine_matrix.rows();
		problem_dim_ = is_scalar ? 1 : dim;

		Eigen::MatrixXd V = extract_vertices(mesh_nodes) * affine_matrix.inverse().transpose();
		auto boundary_nodes = mesh_nodes->boundary_nodes();

		RowVectorNd min = V.colwise().minCoeff();
		RowVectorNd max = V.colwise().maxCoeff();
		const double bbox_size = (max - min).maxCoeff();
		for (int d = 0; d < dim; d++)
			affine_matrix_.col(d) *= max(d) - min(d);
		const double eps = tol * bbox_size;

		index_map_.setConstant(n_bases, 1, -1);

		dependent_map_.resize(n_bases);
		dependent_map_.setConstant(-1);

		int n_pairs = 0;

		// find corresponding periodic boundary nodes
		for (int d = 0; d < dim; d++)
		{
			std::set<int> dependent_ids, target_ids;
			for (int bid : boundary_nodes)
			{
				if (V(bid, d) > max(d) - eps)
					target_ids.insert(bid);
				else if (V(bid, d) < min(d) + eps)
					dependent_ids.insert(bid);
			}

			for (int i : dependent_ids)
			{
				bool found_target = false;
				for (int j : target_ids)
				{
					RowVectorNd projected_diff = V.row(j) - V.row(i);
					projected_diff(d) = 0;
					if (projected_diff.norm() < eps)
					{
						dependent_map_(i) = j;
						n_pairs++;
						found_target = true;
						break;
					}
				}
				if (!found_target)
					log_and_throw_error("Periodic boundary condition failed to find corresponding nodes!");
			}
		}

		{
			periodic_mask_.setZero(n_bases);
			for (int i = 0; i < dependent_map_.size(); i++)
			{
				if (dependent_map_(i) >= 0)
				{
					periodic_mask_(dependent_map_(i)) = 1;
					periodic_mask_(i) = 1;
				}
			}
		}

		// break dependency chains into direct dependency
		for (int d = 0; d < dim; d++)
			for (int i = 0; i < dependent_map_.size(); i++)
				if (dependent_map_(i) >= 0 && dependent_map_(dependent_map_(i)) >= 0)
					dependent_map_(i) = dependent_map_(dependent_map_(i));

		for (int i = 0; i < dependent_map_.size(); i++)
		{
			if (dependent_map_(i) >= 0)
				constraint_list_.push_back({{i, dependent_map_(i)}});
		}

		// new indexing for independent dof
		int independent_dof = 0;
		for (int i = 0; i < dependent_map_.size(); i++)
		{
			if (dependent_map_(i) < 0)
				index_map_(i) = independent_dof++;
		}

		for (int i = 0; i < dependent_map_.size(); i++)
			if (dependent_map_(i) >= 0)
				index_map_(i) = index_map_(dependent_map_(i));

		const int old_size = index_map_.size();
		full_to_periodic_map_.setZero(old_size * problem_dim_);
		for (int i = 0; i < old_size; i++)
			for (int d = 0; d < problem_dim_; d++)
				full_to_periodic_map_(i * problem_dim_ + d) = index_map_(i) * problem_dim_ + d;
	}

	int PeriodicBoundary::full_to_periodic(StiffnessMatrix &A) const
	{
		const int independent_dof = full_to_periodic_map_.maxCoeff() + 1;

		// account for potential pressure block
		auto extended_index_map = [&](int id) {
			if (id < full_to_periodic_map_.size())
				return full_to_periodic_map_(id);
			else
				return (int)(id + independent_dof - full_to_periodic_map_.size());
		};

		StiffnessMatrix A_periodic(extended_index_map(A.rows()), extended_index_map(A.cols()));
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(A.nonZeros());
		for (int k = 0; k < A.outerSize(); k++)
		{
			for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
			{
				entries.emplace_back(extended_index_map(it.row()), extended_index_map(it.col()), it.value());
			}
		}
		A_periodic.setFromTriplets(entries.begin(), entries.end());

		std::swap(A_periodic, A);

		return independent_dof;
	}
	Eigen::MatrixXd PeriodicBoundary::full_to_periodic(const Eigen::MatrixXd &b, bool accumulate) const
	{
		const int independent_dof = full_to_periodic_map_.maxCoeff() + 1;

		// account for potential pressure block
		auto extended_index_map = [&](int id) {
			if (id < full_to_periodic_map_.size())
				return full_to_periodic_map_(id);
			else
				return (int)(id + independent_dof - full_to_periodic_map_.size());
		};

		// rhs under periodic basis
		Eigen::MatrixXd b_periodic;
		b_periodic.setZero(extended_index_map(b.rows()), b.cols());
		if (accumulate)
			for (int k = 0; k < b.rows(); k++)
				b_periodic.row(extended_index_map(k)) += b.row(k);
		else
			for (int k = 0; k < b.rows(); k++)
				b_periodic.row(extended_index_map(k)) = b.row(k);

		return b_periodic;
	}
	std::vector<int> PeriodicBoundary::full_to_periodic(const std::vector<int> &boundary_nodes) const
	{
		std::vector<int> boundary_nodes_reduced = boundary_nodes;
		if (has_periodic_bc())
		{
			for (int i = 0; i < boundary_nodes_reduced.size(); i++)
				boundary_nodes_reduced[i] = full_to_periodic_map_(boundary_nodes_reduced[i]);

			std::sort(boundary_nodes_reduced.begin(), boundary_nodes_reduced.end());
			auto it = std::unique(boundary_nodes_reduced.begin(), boundary_nodes_reduced.end());
			boundary_nodes_reduced.resize(std::distance(boundary_nodes_reduced.begin(), it));
		}
		return boundary_nodes_reduced;
	}

	Eigen::MatrixXd PeriodicBoundary::periodic_to_full(const int ndofs, const Eigen::MatrixXd &x_periodic) const
	{
		const int independent_dof = full_to_periodic_map_.maxCoeff() + 1;

		auto extended_index_map = [&](int id) {
			if (id < full_to_periodic_map_.size())
				return full_to_periodic_map_(id);
			else
				return (int)(id + independent_dof - full_to_periodic_map_.size());
		};

		Eigen::MatrixXd x_full;
		x_full.resize(ndofs, x_periodic.cols());
		for (int i = 0; i < x_full.rows(); i++)
			x_full.row(i) = x_periodic.row(extended_index_map(i));

		return x_full;
	}

	bool PeriodicBoundary::all_direction_periodic() const
	{
		return true;
	}
	bool PeriodicBoundary::has_periodic_bc() const
	{
		return true;
	}
} // namespace polyfem::utils