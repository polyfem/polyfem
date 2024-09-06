#include "MassMatrixAssembler.hpp"

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/ClipperUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <SimpleBVH/BVH.hpp>

namespace polyfem
{
	using namespace basis;
	using namespace quadrature;
	using namespace utils;

	namespace assembler
	{
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
				Quadrature quadrature;

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

			// TODO: use existing PolyFEM code instead of hard coding these gmappings
			VectorNd P1_2D_gmapping(
				const Eigen::MatrixXd &nodes, const Eigen::Vector2d &uv)
			{
				assert(nodes.rows() == 3 && nodes.cols() == 2);
				return (1 - uv[0] - uv[1]) * nodes.row(0) + uv[0] * nodes.row(1) + uv[1] * nodes.row(2);
			}

			VectorNd P1_3D_gmapping(
				const Eigen::MatrixXd &nodes, const Eigen::Vector3d &uvw)
			{
				assert(nodes.rows() == 4 && nodes.cols() == 3);
				return (1 - uvw[0] - uvw[1] - uvw[2]) * nodes.row(0) + uvw[0] * nodes.row(1) + uvw[1] * nodes.row(2) + uvw[2] * nodes.row(3);
			}
		} // namespace

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
			// logger().debug("buffer_size {}", buffer_size);

			mass.resize(n_to_basis * size, n_from_basis * size);
			mass.setZero();

			// auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, mass.rows()));

			Quadrature quadrature;
			if (is_volume)
				TetQuadrature().get_quadrature(2, quadrature);
			else
				TriQuadrature().get_quadrature(2, quadrature);

			// Use a AABB tree to find all intersecting elements then loop over only those pairs
			std::vector<std::array<Eigen::Vector3d, 2>> boxes(from_bases.size());
			for (int i = 0; i < from_bases.size(); i++)
			{
				const Eigen::MatrixXd from_nodes = from_bases[i].nodes();
				boxes[i][0].setZero();
				boxes[i][0].head(size) = from_nodes.colwise().minCoeff();
				boxes[i][1].setZero();
				boxes[i][1].head(size) = from_nodes.colwise().maxCoeff();
			}

			SimpleBVH::BVH bvh;
			bvh.init(boxes);

			// maybe_parallel_for(n_to_basis, [&](int start, int end, int thread_id) {
			// LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);

			std::vector<Eigen::Triplet<double>> triplets;

			for (const ElementBases &to_element : to_bases)
			{
				const Eigen::MatrixXd to_nodes = to_element.nodes();

				std::vector<unsigned int> candidates;
				{
					Eigen::Vector3d bbox_min = Eigen::Vector3d::Zero();
					bbox_min.head(size) = to_nodes.colwise().minCoeff();
					Eigen::Vector3d bbox_max = Eigen::Vector3d::Zero();
					bbox_max.head(size) = to_nodes.colwise().maxCoeff();
					bvh.intersect_box(bbox_min, bbox_max, candidates);
				}

				// for (const ElementBases &from_element : from_bases)
				for (const unsigned int from_element_i : candidates)
				{
					const ElementBases &from_element = from_bases[from_element_i];
					const Eigen::MatrixXd from_nodes = from_element.nodes();

					// Compute the overlap between the two elements as a list of simplices.
					const std::vector<Eigen::MatrixXd> overlap =
						is_volume
							? TetrahedronClipping::clip(to_nodes, from_nodes)
							: TriangleClipping::clip(to_nodes, from_nodes);

					for (const Eigen::MatrixXd &simplex : overlap)
					{
						const double volume = abs(is_volume ? tetrahedron_volume(simplex) : triangle_area(simplex));
						if (abs(volume) == 0.0)
							continue;
						assert(volume > 0);

						for (int qi = 0; qi < quadrature.size(); qi++)
						{
							// NOTE: the 2/6 is neccesary here because the mass matrix assembly use the
							//       determinant of the Jacobian (i.e., area of the parallelogram/volume of the hexahedron)
							const double w = (is_volume ? 6 : 2) * volume * quadrature.weights[qi];
							const VectorNd q = quadrature.points.row(qi);

							const VectorNd p = is_volume ? P1_3D_gmapping(simplex, q) : P1_2D_gmapping(simplex, q);

							// NOTE: Row vector because evaluate_bases expects a rows of a matrix.
							const RowVectorNd from_bc = barycentric_coordinates(p, from_nodes).tail(size).transpose();
							const RowVectorNd to_bc = barycentric_coordinates(p, to_nodes).tail(size).transpose();

							std::vector<AssemblyValues> from_phi, to_phi;
							from_element.evaluate_bases(from_bc, from_phi);
							to_element.evaluate_bases(to_bc, to_phi);

#ifndef NDEBUG
							Eigen::MatrixXd debug;
							from_element.eval_geom_mapping(from_bc, debug);
							assert((debug.transpose() - p).norm() < 1e-12);
							to_element.eval_geom_mapping(to_bc, debug);
							assert((debug.transpose() - p).norm() < 1e-12);
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

			mass.setFromTriplets(triplets.begin(), triplets.end());
			mass.makeCompressed();
		}

	} // namespace assembler
} // namespace polyfem
