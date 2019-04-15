#pragma once

#include <Eigen/Dense>

namespace polyfem
{

	class RefElementSampler
	{
	public:
		static RefElementSampler &sampler();

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


		void sample_polygon(const Eigen::MatrixXd &poly, Eigen::MatrixXd &pts, Eigen::MatrixXi &faces) const;

	private:
		RefElementSampler() { }
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

}
