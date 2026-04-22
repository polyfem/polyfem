#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>

#include <vector>

namespace polyfem
{
	namespace basis
	{
		class ElementBases;
	}

	namespace mesh
	{
		class Mesh;
		class MeshNodes;
		class LocalBoundary;
	} // namespace mesh
} // namespace polyfem

namespace polyfem::utils
{
	class PeriodicBoundary
	{
	public:
		PeriodicBoundary(
			const bool is_scalar,
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::shared_ptr<mesh::MeshNodes> &mesh_nodes,
			const Eigen::MatrixXd &affine_matrix,
			const double tol);

		int full_to_periodic(StiffnessMatrix &A) const;
		Eigen::MatrixXd full_to_periodic(const Eigen::MatrixXd &b, bool accumulate) const;
		std::vector<int> full_to_periodic(const std::vector<int> &boundary_nodes) const;

		inline int dim() const { return problem_dim_; }
		inline int n_periodic_dof() const { return full_to_periodic_map_.maxCoeff() + 1; }
		inline bool is_periodic_dof(const int idx) const { return periodic_mask_[idx]; }

		Eigen::MatrixXd periodic_to_full(const int ndofs, const Eigen::MatrixXd &x_periodic) const;

		bool all_direction_periodic() const;
		bool has_periodic_bc() const;

		int n_constraints() const { return constraint_list_.size(); }
		std::array<int, 2> constraint(const int idx) const { return constraint_list_[idx]; }
		const Eigen::VectorXi &index_map() const { return index_map_; }

		Eigen::MatrixXd get_affine_matrix() const { return affine_matrix_; }

	private:
		int problem_dim_;
		Eigen::VectorXi index_map_;
		Eigen::VectorXi dependent_map_;
		Eigen::VectorXi full_to_periodic_map_;
		Eigen::VectorXi periodic_mask_;
		std::vector<std::array<int, 2>> constraint_list_;

		Eigen::MatrixXd affine_matrix_; // each column is one periodic direction
	};
} // namespace polyfem::utils