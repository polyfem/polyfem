// #include <polyfem/utils/SutherlandHodgmanClipping.hpp>

// #include <Eigen/LU>
// #include <Eigen/SparseCore>

// #include <ipc/broad_phase/aabb.hpp>

// namespace polyfem::utils
// {
// 	/// Compute barycentric coordinates (u, v, w) for point p with respect to triangle (a, b, c).
// 	Eigen::Vector3d barycentric_coordinates(
// 		const Eigen::Vector2d &p,
// 		const Eigen::Vector2d &a,
// 		const Eigen::Vector2d &b,
// 		const Eigen::Vector2d &c)
// 	{
// 		Eigen::Matrix3d A;
// 		A << a[0], b[0], c[0],
// 			a[1], b[1], c[1],
// 			1.0, 1.0, 1.0;
// 		const Eigen::Vector3d rhs(p[0], p[1], 1.0);
// 		const Eigen::Vector3d uvw = A.partialPivLu().solve(rhs);
// 		assert((A * uvw - rhs).norm() < 1e-12);
// 		return uvw;
// 	}

// 	ipc::AABB nodes_to_aabb(const Eigen::MatrixXd &nodes)
// 	{
// 		return ipc::AABB(nodes.rowwise().minCoeff(), nodes.rowwise().maxCoeff());
// 	}

// 	std::vector<std::array<Eigen::Vector2d, 3>> triangle_fan(const std::vector<Eigen::Vector2d> &convex_polygon)
// 	{
// 		assert(convex_polygon.size() >= 3);
// 		std::vector<std::array<Eigen::Vector2d, 3>> triangles;
// 		for (int i = 1; i < convex_polygon.size() - 1; ++i)
// 		{
// 			triangles.push_back(
// 				{{convex_polygon[0], convex_polygon[i], convex_polygon[i + 1]}});
// 		}
// 		return triangles;
// 	}

// 	double triangle_area(const Eigen::Vector2d &a, const Eigen::Vector2d &b, const Eigen::Vector2d &c)
// 	{
// 		Eigen::Matrix3d A;
// 		A << a[0], a[1], 1.0,
// 			b[0], b[1], 1.0,
// 			c[0], c[1], 1.0;
// 		return 0.5 * A.determinant();
// 	}

// 	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
// 	void L2_projection(
// 		const bool is_volume,
// 		const int size,
// 		const int n_basis_a,
// 		const std::vector<ElementBases> &bases_a,
// 		const std::vector<ElementBases> &gbases_a,
// 		const int n_basis_b,
// 		const std::vector<ElementBases> &bases_b,
// 		const std::vector<ElementBases> &gbases_b,
// 		const AssemblyValsCache &cache,
// 		const Eigen::VectorXd &u,
// 		const Eigen::VectorXd &u_proj)
// 	{
// 		MassMatrixAssembler assembler;

// 		Eigen::SparseMatrix<double> M;
// 		assembler.assemble(
// 			is_volume, size, n_basis_b, density, bases_b, gbases_b, cache, M);

// 		Eigen::SparseMatrix<double> A;
// 		assembler.assemble_cross(
// 			is_volume, size,
// 			n_basis_a, bases_a, gbases_a,
// 			n_basis_b, bases_b, gbases_b,
// 			cache, A);

// 		u_proj = M.ldlt().solve(A * u);
// 		assert(np.linalg.norm(M * u_proj - A * u) < 1e-12);

// 		return u_proj;
// 	}

// } // namespace polyfem::utils