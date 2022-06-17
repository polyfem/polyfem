#include "RefElementSampler.hpp"
#include <polyfem/mesh/MeshUtils.hpp>

#include <igl/edges.h>
#include <igl/upsample.h>

#include <cassert>

namespace polyfem
{
	using namespace mesh;
	namespace utils
	{

		///
		/// Generate a canonical triangle/quad subdivided from a regular grid
		///
		/// @param[in]  n  			  n grid quads
		/// @param[in]  tri			  is a tri or a quad
		/// @param[out] V             #V x 2 output vertices positions
		/// @param[out] F             #F x 3 output triangle indices
		///
		void regular_2d_grid(const int n, bool tri, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
		{

			V.resize(n * n, 2);
			F.resize((n - 1) * (n - 1) * 2, 3);
			const double delta = 1. / (n - 1.);
			std::vector<int> map(n * n, -1);

			int index = 0;
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					if (tri && i + j >= n)
						continue;
					map[i + j * n] = index;
					V.row(index) << i * delta, j * delta;
					++index;
				}
			}

			V.conservativeResize(index, 2);

			std::array<int, 3> tmp;

			index = 0;
			for (int i = 0; i < n - 1; ++i)
			{
				for (int j = 0; j < n - 1; ++j)
				{
					tmp = {{map[i + j * n], map[i + 1 + j * n], map[i + (j + 1) * n]}};
					if (tmp[0] >= 0 && tmp[1] >= 0 && tmp[2] >= 0)
					{
						F.row(index) << tmp[0], tmp[1], tmp[2];
						++index;
					}

					tmp = {{map[i + 1 + j * n], map[i + 1 + (j + 1) * n], map[i + (j + 1) * n]}};
					if (tmp[0] >= 0 && tmp[1] >= 0 && tmp[2] >= 0)
					{
						F.row(index) << tmp[0], tmp[1], tmp[2];
						++index;
					}
				}
			}

			F.conservativeResize(index, 3);
		}

		namespace
		{

			void add_tet(const std::array<int, 4> &tmp, const Eigen::MatrixXd &V, int &index, Eigen::MatrixXi &T)
			{
				if (tmp[0] >= 0 && tmp[1] >= 0 && tmp[2] >= 0 && tmp[3] >= 0)
				{
					const Eigen::Vector3d e0 = V.row(tmp[1]) - V.row(tmp[0]);
					const Eigen::Vector3d e1 = V.row(tmp[2]) - V.row(tmp[0]);
					const Eigen::Vector3d e2 = V.row(tmp[3]) - V.row(tmp[0]);
					double vol = (e0.cross(e1)).dot(e2);
					if (vol < 0)
						T.row(index) << tmp[0], tmp[1], tmp[2], tmp[3];
					else
						T.row(index) << tmp[0], tmp[1], tmp[3], tmp[2];
#ifndef NDEBUG
					const Eigen::Vector3d ed0 = V.row(T(index, 1)) - V.row(T(index, 0));
					const Eigen::Vector3d ed1 = V.row(T(index, 2)) - V.row(T(index, 0));
					const Eigen::Vector3d ed2 = V.row(T(index, 3)) - V.row(T(index, 0));
					assert((ed0.cross(ed1)).dot(ed2) < 0);
#endif
					++index;
				}
			}

		} // anonymous namespace

		///
		/// Generate a canonical tet/hex subdivided from a regular grid
		///
		/// @param[in]  n  			  n grid quads
		/// @param[in]  tet			  is a tet or a hex
		/// @param[out] V             #V x 3 output vertices positions
		/// @param[out] F             #F x 3 output triangle indices
		/// @param[out] T             #F x 4 output tet indices
		///
		void regular_3d_grid(const int nn, bool tet, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &T)
		{
			const int n = nn;
			const double delta = 1. / (n - 1.);
			T.resize((n - 1) * (n - 1) * (n - 1) * 6, 4);
			V.resize(n * n * n, 3);
			std::vector<int> map(n * n * n, -1);

			int index = 0;
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					for (int k = 0; k < n; ++k)
					{
						if (tet && i + j + k >= n)
							continue;
						map[(i + j * n) * n + k] = index;
						V.row(index) << i * delta, j * delta, k * delta;
						++index;
					}
				}
			}
			V.conservativeResize(index, 3);

			std::array<int, 8> indices;
			std::array<int, 4> tmp;
			index = 0;
			for (int i = 0; i < n - 1; ++i)
			{
				for (int j = 0; j < n - 1; ++j)
				{
					for (int k = 0; k < n - 1; ++k)
					{
						indices = {{(i + j * n) * n + k,
									(i + 1 + j * n) * n + k,
									(i + 1 + (j + 1) * n) * n + k,
									(i + (j + 1) * n) * n + k,

									(i + j * n) * n + k + 1,
									(i + 1 + j * n) * n + k + 1,
									(i + 1 + (j + 1) * n) * n + k + 1,
									(i + (j + 1) * n) * n + k + 1}};

						tmp = {{map[indices[1 - 1]], map[indices[2 - 1]], map[indices[4 - 1]], map[indices[5 - 1]]}};
						add_tet(tmp, V, index, T);

						tmp = {{map[indices[6 - 1]], map[indices[3 - 1]], map[indices[7 - 1]], map[indices[8 - 1]]}};
						add_tet(tmp, V, index, T);

						tmp = {{map[indices[5 - 1]], map[indices[2 - 1]], map[indices[6 - 1]], map[indices[4 - 1]]}};
						add_tet(tmp, V, index, T);

						tmp = {{map[indices[5 - 1]], map[indices[4 - 1]], map[indices[8 - 1]], map[indices[6 - 1]]}};
						add_tet(tmp, V, index, T);

						tmp = {{map[indices[4 - 1]], map[indices[2 - 1]], map[indices[6 - 1]], map[indices[3 - 1]]}};
						add_tet(tmp, V, index, T);

						tmp = {{map[indices[3 - 1]], map[indices[4 - 1]], map[indices[8 - 1]], map[indices[6 - 1]]}};
						add_tet(tmp, V, index, T);
					}
				}
			}

			T.conservativeResize(index, 4);

			F.resize(4 * index, 3);

			F.block(0, 0, index, 1) = T.col(1);
			F.block(0, 1, index, 1) = T.col(0);
			F.block(0, 2, index, 1) = T.col(2);

			F.block(index, 0, index, 1) = T.col(0);
			F.block(index, 1, index, 1) = T.col(1);
			F.block(index, 2, index, 1) = T.col(3);

			F.block(2 * index, 0, index, 1) = T.col(1);
			F.block(2 * index, 1, index, 1) = T.col(2);
			F.block(2 * index, 2, index, 1) = T.col(3);

			F.block(3 * index, 0, index, 1) = T.col(2);
			F.block(3 * index, 1, index, 1) = T.col(0);
			F.block(3 * index, 2, index, 1) = T.col(3);
		}

		void RefElementSampler::init(const bool is_volume, const int n_elements, double target_rel_area)
		{
			is_volume_ = is_volume;

			area_param_ = target_rel_area * n_elements;
#ifndef NDEBUG
			area_param_ *= 10.0;
#endif

			build();
		}

		void RefElementSampler::build()
		{
			using namespace Eigen;

			if (is_volume_)
			{
				{
					MatrixXd pts(8, 3);
					pts << 0, 0, 0,
						0, 1, 0,
						1, 1, 0,
						1, 0, 0,

						//4
						0, 0, 1,
						0, 1, 1,
						1, 1, 1,
						1, 0, 1;

					Eigen::MatrixXi faces(12, 3);
					faces << 1, 2, 0,
						0, 2, 3,

						5, 4, 6,
						4, 7, 6,

						1, 0, 4,
						1, 4, 5,

						2, 1, 5,
						2, 5, 6,

						3, 2, 6,
						3, 6, 7,

						0, 3, 7,
						0, 7, 4;

					regular_3d_grid(std::max(2., round(1. / pow(area_param_, 1. / 3.) + 1) / 2.), false, cube_points_, cube_faces_, cube_tets_);

					// Extract sampled edges matching the base element edges
					Eigen::MatrixXi edges(12, 2);
					edges << 0, 1,
						1, 2,
						2, 3,
						3, 0,

						4, 5,
						5, 6,
						6, 7,
						7, 4,

						0, 4,
						1, 5,
						2, 6,
						3, 7;
					igl::edges(cube_faces_, cube_edges_);
					extract_parent_edges(cube_points_, cube_edges_, pts, edges, cube_edges_);

					// Same local order as in FEMBasis3d
					cube_corners_.resize(8, 3);
					cube_corners_ << 0, 0, 0,
						1, 0, 0,
						1, 1, 0,
						0, 1, 0,
						0, 0, 1,
						1, 0, 1,
						1, 1, 1,
						0, 1, 1;
				}
				{
					MatrixXd pts(4, 3);
					pts << 0, 0, 0,
						1, 0, 0,
						0, 1, 0,
						0, 0, 1;

					Eigen::MatrixXi faces(4, 3);
					faces << 0, 1, 2,

						3, 1, 0,
						2, 1, 3,
						0, 2, 3;

					regular_3d_grid(std::max(2., round(1. / pow(area_param_, 1. / 3.) + 1)), true, simplex_points_, simplex_faces_, simplex_tets_);

					// Extract sampled edges matching the base element edges
					Eigen::MatrixXi edges;
					igl::edges(faces, edges);
					igl::edges(simplex_faces_, simplex_edges_);
					extract_parent_edges(simplex_points_, simplex_edges_, pts, edges, simplex_edges_);

					// Same local order as in FEMBasis3d
					simplex_corners_.resize(4, 3);
					simplex_corners_ << 0, 0, 0,
						1, 0, 0,
						0, 1, 0,
						0, 0, 1;
				}
			}
			else
			{
				{
					MatrixXd pts(4, 2);
					pts << 0, 0,
						0, 1,
						1, 1,
						1, 0;

					MatrixXi E(4, 2);
					E << 0, 1,
						1, 2,
						2, 3,
						3, 0;

					regular_2d_grid(std::max(2., round(1. / sqrt(area_param_) + 1)), false, cube_points_, cube_faces_);

					// Extract sampled edges matching the base element edges
					igl::edges(cube_faces_, cube_edges_);
					extract_parent_edges(cube_points_, cube_edges_, pts, E, cube_edges_);

					// Same local order as in FEMBasis2d
					cube_corners_.resize(4, 2);
					cube_corners_ << 0, 0,
						1, 0,
						1, 1,
						0, 1;
				}
				{
					MatrixXd pts(3, 2);
					pts << 0, 0,
						0, 1,
						1, 0;

					MatrixXi E(3, 2);
					E << 0, 1,
						1, 2,
						2, 0;

					regular_2d_grid(std::max(2., round(1. / sqrt(area_param_) + 1)), true, simplex_points_, simplex_faces_);

					// Extract sampled edges matching the base element edges
					igl::edges(simplex_faces_, simplex_edges_);
					extract_parent_edges(simplex_points_, simplex_edges_, pts, E, simplex_edges_);

					// Same local order as in FEMBasis2d
					simplex_corners_.resize(3, 2);
					simplex_corners_ << 0, 0,
						1, 0,
						0, 1;
				}
			}
		}

		void RefElementSampler::sample_polygon(const Eigen::MatrixXd &poly, Eigen::MatrixXd &pts, Eigen::MatrixXi &faces) const
		{
			pts.resize(poly.rows() + 1, poly.cols());
			pts.block(0, 0, poly.rows(), poly.cols()) = poly;
			pts.row(poly.rows()) = poly.colwise().mean();

			faces.resize(poly.rows(), 3);

			double area_avg = 0;

			for (int e = 0; e < int(poly.rows()); ++e)
			{
				faces.row(e) << e, (e + 1) % poly.rows(), poly.rows();

				Eigen::Matrix2d tmp;
				tmp.row(0) = pts.row(e) - pts.row(poly.rows());
				tmp.row(1) = pts.row((e + 1) % poly.rows()) - pts.row(poly.rows());
				area_avg += fabs(tmp.determinant());
			}

			area_avg /= poly.rows();

			const int n_refs = area_avg / area_param_ * 40;

			Eigen::MatrixXd new_pts;
			Eigen::MatrixXi new_faces;

			igl::upsample(pts, faces, new_pts, new_faces, n_refs);

			pts = new_pts;
			faces = new_faces;
		}

		void RefElementSampler::sample_polyhedron(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &f, Eigen::MatrixXd &pts, Eigen::MatrixXi &faces) const
		{
			//TODO
			pts = vertices;
			faces = f;
		}
	} // namespace utils
} // namespace polyfem
