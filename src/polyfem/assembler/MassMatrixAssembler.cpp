#include "MassMatrixAssembler.hpp"

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/ClipperUtils.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	using namespace polyfem::basis;
	using namespace polyfem::quadrature;
	using namespace polyfem::utils;

	namespace
	{
		class LocalThreadMatStorage
		{
		public:
			std::vector<Eigen::Triplet<double>> entries;
			StiffnessMatrix tmp_mat;
			StiffnessMatrix mass_mat;
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadMatStorage()
			{
			}

			LocalThreadMatStorage(const int buffer_size, const int mat_size)
			{
				init(buffer_size, mat_size);
			}

			void init(const int buffer_size, const int mat_size)
			{
				entries.reserve(buffer_size);
				tmp_mat.resize(mat_size, mat_size);
				mass_mat.resize(mat_size, mat_size);
			}

			void condense()
			{
				if (entries.size() >= 1e8)
				{
					tmp_mat.setFromTriplets(entries.begin(), entries.end());
					mass_mat += tmp_mat;
					mass_mat.makeCompressed();

					tmp_mat.setZero();
					tmp_mat.data().squeeze();

					mass_mat.makeCompressed();

					entries.clear();
					logger().debug("cleaning memory...");
				}
			}
		};
	} // namespace

	void MassMatrixAssembler::assemble(
		const bool is_volume,
		const int size,
		const int n_basis,
		const Density &density,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		StiffnessMatrix &mass) const
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * size);
		logger().debug("buffer_size {}", buffer_size);

		mass.resize(n_basis * size, n_basis * size);
		mass.setZero();

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, mass.rows()));

		const int n_bases = int(bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				ElementAssemblyValues &vals = local_storage.vals;
				// vals.compute(e, is_volume, bases[e], gbases[e]);
				cache.compute(e, is_volume, bases[e], gbases[e], vals);

				const Quadrature &quadrature = vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = vals.det.array() * quadrature.weights.array();
				const int n_loc_bases = int(vals.basis_values.size());

				for (int i = 0; i < n_loc_bases; ++i)
				{
					const auto &global_i = vals.basis_values[i].global;

					for (int j = 0; j <= i; ++j)
					{
						const auto &global_j = vals.basis_values[j].global;

						double tmp = 0; //(vals.basis_values[i].val.array() * vals.basis_values[j].val.array() * da.array()).sum();
						for (int q = 0; q < local_storage.da.size(); ++q)
						{
							const double rho = density(vals.quadrature.points.row(q), vals.val.row(q), vals.element_id);
							tmp += rho * vals.basis_values[i].val(q) * vals.basis_values[j].val(q) * local_storage.da(q);
						}
						if (std::abs(tmp) < 1e-30)
						{
							continue;
						}

						for (int n = 0; n < size; ++n)
						{
							// local matrix is diagonal
							const int m = n;
							// for(int m = 0; m < size; ++m)
							{
								const double local_value = tmp; // val(n*size+m);
								for (size_t ii = 0; ii < global_i.size(); ++ii)
								{
									const auto gi = global_i[ii].index * size + m;
									const auto wi = global_i[ii].val;

									for (size_t jj = 0; jj < global_j.size(); ++jj)
									{
										const auto gj = global_j[jj].index * size + n;
										const auto wj = global_j[jj].val;

										local_storage.entries.emplace_back(gi, gj, local_value * wi * wj);
										if (j < i)
										{
											local_storage.entries.emplace_back(gj, gi, local_value * wj * wi);
										}

										local_storage.condense();
									}
								}
							}
						}

						// t1.stop();
						// if (!vals.has_parameterization) { std::cout << "-- t1: " << t1.getElapsedTime() << std::endl; }
					}
				}

				// timer.stop();
				// if (!vals.has_parameterization) { std::cout << "-- Timer: " << timer.getElapsedTime() << std::endl; }
			}
		});

		// Serially merge local storages
		for (LocalThreadMatStorage &local_storage : storage)
		{
			mass += local_storage.mass_mat;
			local_storage.tmp_mat.setFromTriplets(local_storage.entries.begin(), local_storage.entries.end());
			mass += local_storage.tmp_mat;
		}
		mass.makeCompressed();
	}

	namespace
	{
		/// Compute barycentric coordinates (u, v, w) for point p with respect to triangle (a, b, c).
		Eigen::Vector3d barycentric_coordinates(
			const Eigen::Vector2d &p,
			const Eigen::Vector2d &a,
			const Eigen::Vector2d &b,
			const Eigen::Vector2d &c)
		{
			Eigen::Matrix3d A;
			A << a[0], b[0], c[0],
				a[1], b[1], c[1],
				1.0, 1.0, 1.0;
			const Eigen::Vector3d rhs(p[0], p[1], 1.0);
			// TODO: Can we use better than LU?
			const Eigen::Vector3d uvw = A.partialPivLu().solve(rhs);
			assert((A * uvw - rhs).norm() / rhs.norm() < 1e-12);
			return uvw;
		}

		std::vector<Eigen::MatrixXd> triangle_fan(const Eigen::MatrixXd &convex_polygon)
		{
			assert(convex_polygon.rows() >= 3);
			std::vector<Eigen::MatrixXd> triangles;
			for (int i = 1; i < convex_polygon.rows() - 1; ++i)
			{
				triangles.emplace_back(3, convex_polygon.cols());
				triangles.back().row(0) = convex_polygon.row(0);
				triangles.back().row(1) = convex_polygon.row(i);
				triangles.back().row(2) = convex_polygon.row(i + 1);
			}
			return triangles;
		}

		double triangle_area(const Eigen::MatrixXd &triangle)
		{
			const auto &a = triangle.row(0);
			const auto &b = triangle.row(1);
			const auto &c = triangle.row(2);
			return 0.5 * ((b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y()));
		}

		Eigen::Vector2d P1_2D_gmapping(
			const Eigen::MatrixXd &nodes, const Eigen::Vector2d &uv)
		{
			assert(nodes.rows() == 3);
			return (1 - uv[0] - uv[1]) * nodes.row(0) + uv[0] * nodes.row(1) + uv[1] * nodes.row(2);
		}

		Eigen::MatrixXd triangle_to_clockwise_order(const Eigen::MatrixXd &T)
		{
			assert(T.rows() == 3 && T.cols() == 2);
			Eigen::Matrix3d A;
			A << T(0, 0), T(0, 1), 1,
				T(1, 0), T(1, 1), 1,
				T(2, 0), T(2, 1), 1;
			if (A.determinant() <= 0)
				return T;

			Eigen::MatrixXd T_clockwise(T.rows(), T.cols());
			T_clockwise.row(0) = T.row(2);
			T_clockwise.row(1) = T.row(1);
			T_clockwise.row(2) = T.row(0);

			return T_clockwise;
		}
	}; // namespace

	void MassMatrixAssembler::assemble_cross(
		const bool is_volume,
		const int size,
		const int n_from_basis,
		const std::vector<basis::ElementBases> &from_bases,
		const std::vector<basis::ElementBases> &from_gbases,
		const int n_to_basis,
		const std::vector<basis::ElementBases> &to_bases,
		const std::vector<basis::ElementBases> &to_gbases,
		const AssemblyValsCache &cache,
		StiffnessMatrix &mass) const
	{
		const int buffer_size = std::min(long(1e8), long(std::max(n_from_basis, n_to_basis)) * size);
		logger().debug("buffer_size {}", buffer_size);

		mass.resize(n_to_basis * size, n_from_basis * size);
		mass.setZero();

		// auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, mass.rows()));

		// TODO: Why are we shadowing this variable?
		// const int n_from_bases = int(from_bases.size());
		// const int n_to_bases = int(to_bases.size());

		// TODO: Use a AABB tree to find all intersecting elements then loop over only those pairs

		// maybe_parallel_for(n_to_basis, [&](int start, int end, int thread_id) {
		// LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);

		std::vector<Eigen::Triplet<double>> triplets;

		Quadrature quadrature;
		TriQuadrature().get_quadrature(2, quadrature);

		// static int i = 0;
		// std::vector<Eigen::Vector2d> vertices;

		for (const ElementBases &to_element : to_bases)
		{
			const Eigen::MatrixXd to_nodes = to_element.nodes();
			assert(to_nodes.rows() == 3);
			const Eigen::MatrixXd to_nodes_clockwise = triangle_to_clockwise_order(to_nodes);

			for (const ElementBases &from_element : from_bases)
			{
				const Eigen::MatrixXd from_nodes = from_element.nodes();
				assert(from_nodes.rows() == 3);
				const Eigen::MatrixXd from_nodes_clockwise = triangle_to_clockwise_order(from_nodes);

				const std::vector<Eigen::MatrixXd> overlaps = PolygonClipping::clip(to_nodes_clockwise, from_nodes_clockwise);
				assert(overlaps.size() <= 1);
				if (overlaps.empty())
					continue;
				const Eigen::MatrixXd &overlap = overlaps[0];

				if (overlap.size() < 3)
					continue;

				const std::vector<Eigen::MatrixXd> triangles = triangle_fan(overlap);

				for (const Eigen::MatrixXd &triangle : triangles)
				{
					const double area = abs(triangle_area(triangle));
					if (abs(area) == 0.0)
						continue;
					assert(area > 0);
					// vertices.emplace_back(triangle.row(0));
					// vertices.emplace_back(triangle.row(1));
					// vertices.emplace_back(triangle.row(2));

					for (int qi = 0; qi < quadrature.size(); qi++)
					{
						// NOTE: the 2 is neccesary here because the mass matrix assembly use the
						//       determinant of the Jacobian (i.e., area of the parallelogram)
						const double w = 2 * area * quadrature.weights[qi];
						const Eigen::Vector2d q = quadrature.points.row(qi);

						const Eigen::Vector2d p = P1_2D_gmapping(triangle, q);

						const Eigen::RowVector2d from_uv =
							barycentric_coordinates(p, from_nodes.row(0), from_nodes.row(1), from_nodes.row(2)).tail<2>().transpose();
						const Eigen::RowVector2d to_uv =
							barycentric_coordinates(p, to_nodes.row(0), to_nodes.row(1), to_nodes.row(2)).tail<2>().transpose();

						std::vector<AssemblyValues> from_phi, to_phi;
						from_element.evaluate_bases(from_uv, from_phi);
						to_element.evaluate_bases(to_uv, to_phi);

#ifndef NDEBUG
						Eigen::MatrixXd debug;
						from_element.eval_geom_mapping(from_uv, debug);
						assert((debug.transpose() - p).norm() < 1e-15);
						to_element.eval_geom_mapping(to_uv, debug);
						assert((debug.transpose() - p).norm() < 1e-15);
#endif

						for (int n = 0; n < size; ++n)
						{
							// local matrix is diagonal
							const int m = n;
							{
								for (int to_local_i = 0; to_local_i < to_phi.size(); ++to_local_i)
								{
									const int to_global_i = to_element.bases[to_local_i].global()[0].index * size + m;
									for (int from_local_i = 0; from_local_i < from_phi.size(); ++from_local_i)
									{
										const auto from_global_i = from_element.bases[from_local_i].global()[0].index * size + n;
										triplets.emplace_back(
											to_global_i, from_global_i,
											w * from_phi[from_local_i].val(0) * to_phi[to_local_i].val(0));
									}
								}
							}
						}
					}
				}
			}
		}

		// open file for writing obj
		// std::ofstream obj_file;
		// obj_file.open(fmt::format("debug_{:03}.obj", i++));
		// for (const auto &vertex : vertices)
		// {
		// 	obj_file << fmt::format("v {:g} {:g} 0\n", vertex(0), vertex(1));
		// }
		// for (int i = 0; i < vertices.size(); i += 3)
		// {
		// 	obj_file << fmt::format("f {} {} {}\n", i + 1, i + 2, i + 3);
		// }
		// obj_file.close();

		mass.setFromTriplets(triplets.begin(), triplets.end());
		mass.makeCompressed();
	}

} // namespace polyfem::assembler
