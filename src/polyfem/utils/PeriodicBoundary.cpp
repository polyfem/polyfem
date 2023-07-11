#include "PeriodicBoundary.hpp"
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::utils
{
    PeriodicBoundary::PeriodicBoundary(
        const mesh::Mesh &mesh,
        const std::vector<mesh::LocalBoundary> &total_local_boundary,
        const bool is_scalar,
        const int n_bases,
        const std::vector<basis::ElementBases> &bases,
        const std::shared_ptr<mesh::MeshNodes> &mesh_nodes,
        const std::vector<bool> &periodic_dimensions): periodic_dimensions_(periodic_dimensions)
    {
        const int dim = mesh.dimension();
        problem_dim_ = is_scalar ? 1 : dim;
		const double eps = 1e-5;

		RowVectorNd min, max;
		mesh.bounding_box(min, max);
		const double bbox_size = (max - min).maxCoeff();

        Eigen::VectorXi index_map;
		index_map.setConstant(n_bases, 1, -1);

		Eigen::VectorXi dependent_map(n_bases);
		dependent_map.setConstant(-1);

		int n_pairs = 0;

		// find corresponding periodic boundary nodes
		Eigen::Vector3i dependent_face({1, 2, 5}), target_face({3, 4, 6});
		for (int d = 0; d < dim; d++)
		{
			if (!periodic_dimensions_[d])
				continue;

			const int dependent_boundary_id = dependent_face(d);
			const int target_boundary_id = target_face(d);

			std::set<int> dependent_ids, target_ids;

			for (const auto &lb : total_local_boundary)
			{
				const int e = lb.element_id();
				const basis::ElementBases &bs = bases[e];

				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh);

					for (long n = 0; n < nodes.size(); ++n)
					{
						for (int j = 0; j < bs.bases[nodes(n)].global().size(); j++)
						{
							const int gid = bs.bases[nodes(n)].global()[j].index;
							if (mesh.get_boundary_id(primitive_global_id) == dependent_boundary_id)
								dependent_ids.insert(gid);
							else if (mesh.get_boundary_id(primitive_global_id) == target_boundary_id)
								target_ids.insert(gid);
						}
					}
				}
			}

			for (int i : dependent_ids)
			{
				bool found_target = false;
				for (int j : target_ids)
				{
					RowVectorNd projected_diff = mesh_nodes->node_position(j) - mesh_nodes->node_position(i);
					projected_diff(d) = 0;
					if (projected_diff.norm() < eps * bbox_size)
					{
						dependent_map(i) = j;
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
			for (int i = 0; i < dependent_map.size(); i++)
			{
				if (dependent_map(i) >= 0)
				{
					periodic_mask_(dependent_map(i)) = 1;
					periodic_mask_(i) = 1;
				}
			}
		}

		// break dependency chains into direct dependency
		for (int d = 0; d < dim; d++)
			for (int i = 0; i < dependent_map.size(); i++)
				if (dependent_map(i) >= 0 && dependent_map(dependent_map(i)) >= 0)
					dependent_map(i) = dependent_map(dependent_map(i));

		// new indexing for independent dof
		int independent_dof = 0;
		for (int i = 0; i < dependent_map.size(); i++)
		{
			if (dependent_map(i) < 0)
				index_map(i) = independent_dof++;
		}

		for (int i = 0; i < dependent_map.size(); i++)
			if (dependent_map(i) >= 0)
				index_map(i) = index_map(dependent_map(i));

        const int old_size = index_map.size();
        full_to_periodic_map_.setZero(old_size * problem_dim_);
        for (int i = 0; i < old_size; i++)
            for (int d = 0; d < problem_dim_; d++)
                full_to_periodic_map_(i * problem_dim_ + d) = index_map(i) * problem_dim_ + d;
    }

    int PeriodicBoundary::full_to_periodic(StiffnessMatrix &A) const
	{
		const int independent_dof = full_to_periodic_map_.maxCoeff() + 1;
		
		// account for potential pressure block
		auto index_map = [&](int id){
			if (id < full_to_periodic_map_.size())
				return full_to_periodic_map_(id);
			else
				return (int)(id + independent_dof - full_to_periodic_map_.size());
		};

		StiffnessMatrix A_periodic(index_map(A.rows()), index_map(A.cols()));
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(A.nonZeros());
		for (int k = 0; k < A.outerSize(); k++)
		{
			for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
			{
				entries.emplace_back(index_map(it.row()), index_map(it.col()), it.value());
			}
		}
		A_periodic.setFromTriplets(entries.begin(),entries.end());

		std::swap(A_periodic, A);

		return independent_dof;
	}
    Eigen::MatrixXd PeriodicBoundary::full_to_periodic(const Eigen::MatrixXd &b, bool accumulate) const
    {
		const int independent_dof = full_to_periodic_map_.maxCoeff() + 1;
		
		// account for potential pressure block
		auto index_map = [&](int id){
			if (id < full_to_periodic_map_.size())
				return full_to_periodic_map_(id);
			else
				return (int)(id + independent_dof - full_to_periodic_map_.size());
		};

		// rhs under periodic basis
		Eigen::MatrixXd b_periodic;
		b_periodic.setZero(index_map(b.rows()), b.cols());
		if (accumulate)
			for (int k = 0; k < b.rows(); k++)
				b_periodic.row(index_map(k)) += b.row(k);
		else
			for (int k = 0; k < b.rows(); k++)
				b_periodic.row(index_map(k)) = b.row(k);

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
		
		auto index_map = [&](int id){
			if (id < full_to_periodic_map_.size())
				return full_to_periodic_map_(id);
			else
				return (int)(id + independent_dof - full_to_periodic_map_.size());
		};

		Eigen::MatrixXd x_full;
		x_full.resize(ndofs, x_periodic.cols());
		for (int i = 0; i < x_full.rows(); i++)
			x_full.row(i) = x_periodic.row(index_map(i));

		return x_full;
	}

    bool PeriodicBoundary::all_direction_periodic() const
    {
        for (const bool &r : periodic_dimensions_)
        {
            if (!r)
                return false;
        }
        return true;
    }
    bool PeriodicBoundary::has_periodic_bc() const
    {
        for (const bool &r : periodic_dimensions_)
        {
            if (r)
                return true;
        }
        return false;
    }
}