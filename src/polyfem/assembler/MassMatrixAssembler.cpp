#include "MassMatrixAssembler.hpp"

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/utils/SutherlandHodgmanClipping.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
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
							//local matrix is diagonal
							const int m = n;
							// for(int m = 0; m < size; ++m)
							{
								const double local_value = tmp; //val(n*size+m);
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
			assert((A * uvw - rhs).norm() < 1e-12);
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
			Eigen::Matrix3d A;
			A.leftCols<2>() = triangle;
			A.col(2).setOnes();
			return 0.5 * A.determinant();
		}

		Eigen::Vector2d P1_2D_gmapping(
			const Eigen::MatrixXd &nodes, const Eigen::Vector2d &uv)
		{
			assert(nodes.rows() == 3);
			return (1 - uv[0] - uv[1]) * nodes.row(0) + uv[0] * nodes.row(1) + uv[1] * nodes.row(2);
		}

		void reverse_rows(Eigen::MatrixXd &A)
		{
			for (int i = 0; i < A.rows() / 2; ++i)
			{
				Eigen::RowVectorXd tmp = A.row(i);
				A.row(i) = A.row(A.rows() - i - 1);
				A.row(A.rows() - i - 1) = tmp;
			}
		}
	}; // namespace

	void MassMatrixAssembler::assemble_cross(
		const bool is_volume,
		const int size,
		const int n_basis_a,
		const std::vector<ElementBases> &bases_a,
		const std::vector<ElementBases> &gbases_a,
		const int n_basis_b,
		const std::vector<ElementBases> &bases_b,
		const std::vector<ElementBases> &gbases_b,
		const AssemblyValsCache &cache,
		StiffnessMatrix &mass) const
	{
		const int buffer_size = std::min(long(1e8), long(std::max(n_basis_a, n_basis_b)) * size);
		logger().debug("buffer_size {}", buffer_size);

		mass.resize(n_basis_b * size, n_basis_a * size);
		mass.setZero();

		// auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, mass.rows()));

		// TODO: Why are we shadowing this variable?
		const int n_bases_a = int(bases_a.size());
		const int n_bases_b = int(bases_b.size());

		// TODO: Use a AABB tree to find all intersecting elements then loop over only those pairs

		// maybe_parallel_for(n_bases_b, [&](int start, int end, int thread_id) {
		// LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);

		std::vector<Eigen::Triplet<double>> triplets;

		Quadrature quadrature;
		TriQuadrature().get_quadrature(2, quadrature);

		for (int ebi = 0; ebi < n_bases_b; ++ebi)
		{
			const ElementBases &eb = bases_b[ebi];
			Eigen::MatrixXd eb_nodes = eb.nodes();
			assert(eb_nodes.rows() == 3);
			reverse_rows(eb_nodes); // clockwise order

			for (int eai = 0; eai < n_bases_a; ++eai)
			{
				const ElementBases &ea = bases_a[eai];
				Eigen::MatrixXd ea_nodes = ea.nodes();
				assert(ea_nodes.rows() == 3);
				reverse_rows(ea_nodes); // clockwise order

				Eigen::MatrixXd overlap = sutherland_hodgman_clipping(eb_nodes, ea_nodes);
				reverse_rows(overlap); // back to counter-clockwise order
				if (overlap.size() < 3)
					continue;
				const std::vector<Eigen::MatrixXd> triangles = triangle_fan(overlap);

				for (const Eigen::MatrixXd &triangle : triangles)
				{
					const double area = triangle_area(triangle);
					if (abs(area) < 1e-12)
						continue;
					assert(area > 0);

					for (int qi = 0; qi < quadrature.size(); qi++)
					{
						const double w = quadrature.weights[qi];
						const Eigen::VectorXd q = quadrature.points.row(qi);

						const Eigen::Vector2d p = P1_2D_gmapping(triangle, q);

						const Eigen::Vector3d x_i = barycentric_coordinates(
							eb_nodes.row(0), eb_nodes.row(1), eb_nodes.row(2), p);
#ifndef NDEBUG
						// eb.eval_geom_mapping(x_i.transpose())
						// assert((p - element_b.gmapping(x_i)).norm() < 1e-12);
#endif

						const Eigen::Vector3d x_j = barycentric_coordinates(
							ea_nodes.row(0), ea_nodes.row(1), ea_nodes.row(2), p);
						// assert((p - element_a.gmapping(x_i)).norm() < 1e-12)

						std::vector<AssemblyValues> phi_i, phi_j;
						eb.evaluate_bases(x_i.head<2>().transpose(), phi_i);
						ea.evaluate_bases(x_j.head<2>().transpose(), phi_j);

						for (int n = 0; n < size; ++n)
						{
							for (int m = 0; m < size; ++m)
							{
								for (int loc_i = 0; loc_i < phi_i.size(); ++loc_i)
								{
									for (int loc_j = 0; loc_j < phi_j.size(); ++loc_j)
									{
										triplets.emplace_back(
											eb.bases[loc_i].global()[0].index * size + n,
											ea.bases[loc_j].global()[0].index * size + n,
											w * phi_i[loc_i].val(0) * phi_j[loc_i].val(0) * area);
									}
								}
							}
						}
					}
				}
			}
		}

		mass.setFromTriplets(triplets.begin(), triplets.end());
		mass.makeCompressed();
	}

} // namespace polyfem::assembler
