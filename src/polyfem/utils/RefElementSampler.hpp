#pragma once

#include <Eigen/Dense>

namespace polyfem
{
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
		void regular_2d_grid(const int n, bool tri, Eigen::MatrixXd &V, Eigen::MatrixXi &F);

		///
		/// Generate a canonical tet/hex subdivided from a regular grid
		///
		/// @param[in]  n  			  n grid quads
		/// @param[in]  tet			  is a tet or a hex
		/// @param[out] V             #V x 3 output vertices positions
		/// @param[out] F             #F x 3 output triangle indices
		/// @param[out] T             #F x 4 output tet indices
		///
		void regular_3d_grid(const int nn, bool tet, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &T);

		class RefElementSampler
		{
		public:
			RefElementSampler() {}
			void init(const bool is_volume, const int n_elements, const double target_rel_area);

			const Eigen::MatrixXd &cube_corners() const { return cube_corners_; }
			const Eigen::MatrixXd &cube_points() const { return cube_points_; }
			const Eigen::MatrixXi &cube_faces() const { return cube_faces_; }
			const Eigen::MatrixXi &cube_volume() const { return is_volume_ ? cube_tets_ : cube_faces_; }
			const Eigen::MatrixXi &cube_edges() const { return cube_edges_; }

			const Eigen::MatrixXd &simplex_corners() const { return simplex_corners_; }
			const Eigen::MatrixXd &simplex_points() const { return simplex_points_; }
			const Eigen::MatrixXi &simplex_faces() const { return simplex_faces_; }
			const Eigen::MatrixXi &simplex_volume() const { return is_volume_ ? simplex_tets_ : simplex_faces_; }
			const Eigen::MatrixXi &simplex_edges() const { return simplex_edges_; }

			void sample_polygon(const Eigen::MatrixXd &poly, Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXi &edges) const;
			void sample_polyhedron(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &f, Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXi &edges) const;

			inline int num_samples() const
			{
				return is_volume_ ? std::max(2., round(1. / pow(area_param_, 1. / 3.) + 1)) : std::max(2., round(1. / sqrt(area_param_) + 1));
			}

		private:
			void build();

			Eigen::MatrixXi cube_tets_;
			Eigen::MatrixXi simplex_tets_;

			Eigen::MatrixXd cube_corners_;
			Eigen::MatrixXd cube_points_;
			Eigen::MatrixXi cube_faces_;
			Eigen::MatrixXi cube_edges_;

			Eigen::MatrixXd simplex_corners_;
			Eigen::MatrixXd simplex_points_;
			Eigen::MatrixXi simplex_faces_;
			Eigen::MatrixXi simplex_edges_;

			double area_param_;
			double is_volume_;
		};
	} // namespace utils
} // namespace polyfem
