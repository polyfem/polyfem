#pragma once

#include <ipc/collision_mesh.hpp>

namespace polyfem
{
	// Enrich the ipc::CollisionMesh class with read-only access to
	// displacement_map and its transpose as needed for fast Hessian assembly.
	// Our caching here assumes that `ipc::CollisionMesh` is indeed immutable as
	// claimed in `ipc/collision_mesh.hpp`.
	class CollisionMesh : public ipc::CollisionMesh
	{
	public:
		using ipc::CollisionMesh::CollisionMesh;

		const Eigen::SparseMatrix<double> &displacement_map() const
		{
			return m_displacement_map;
		}

		const Eigen::SparseMatrix<double> &displacement_map_transpose() const
		{
			if (!m_has_cached_transpose)
			{
				m_displacement_map_transpose = m_displacement_map.transpose();
				m_has_cached_transpose = true;
			}
			return m_displacement_map_transpose;
		}

		template<class F>
		void visit_displacement_map_row(int row, F &&f) const
		{
			assert(0 <= row && row < m_displacement_map.rows());
			const auto &dmt = displacement_map_transpose();
			for (Eigen::SparseMatrix<double>::InnerIterator it(dmt, row); it; ++it)
			{
				f(it.row(), it.value());
			}
		}

		// Whether there is a nontrivial displacement map, e.g. due to
		// the use of a proxy mesh. In this case we cannot
		// use `to_full_vertex_id` directly in assembly and must iterate
		// over rows of the displacement map.
		bool has_nontrivial_displacement_map() const
		{
			if (m_has_nontrivial_displacement_map == -1) {
				m_has_nontrivial_displacement_map =
						m_displacement_map.rows() != m_select_vertices.rows()
					||  m_displacement_map.cols() != m_select_vertices.cols()
					||  m_displacement_map.nonZeros() != m_select_vertices.nonZeros();
				if (!m_has_nontrivial_displacement_map) {
					// TODO: more efficient comparison that avoids sparse matrix
					// subtraction.
					// Note that this check does need to be guarded by the size
					// checks above to prevent `isApprox` from throwing an
					// assertion.
					m_has_nontrivial_displacement_map = !(m_displacement_map.isApprox(m_select_vertices, 0));
				}
			}

			return (m_has_nontrivial_displacement_map != 0);
		}

	private:
		mutable Eigen::SparseMatrix<double> m_displacement_map_transpose;
		mutable bool m_has_cached_transpose = false;
		mutable int m_has_nontrivial_displacement_map = -1;
	};
} // namespace polyfem
