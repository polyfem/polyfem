#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/GenericProblem.hpp>
#include <polyfem/ExpressionValue.hpp>

#include <Eigen/Dense>

namespace polyfem
{
	class Obstacle
	{
	public:
		void init(const json &json, const std::string &root_path, const int dim);

		inline int n_vertices() const { return v_.rows(); }
		inline const Eigen::MatrixXd &v() const { return v_; }
		inline const Eigen::VectorXi &codim_v() const { return codim_v_; }
		inline const Eigen::MatrixXi &f() const { return f_; }
		inline const Eigen::MatrixXi &e() const { return e_; }
		inline const Eigen::MatrixXi &f_2_e() const { return f_2_e_; }

		inline const Eigen::MatrixXi &get_face_connectivity() const { return in_f_; }
		inline const Eigen::MatrixXi &get_edge_connectivity() const { return in_e_; }
		inline const Eigen::VectorXi &get_vertex_connectivity() const { return in_v_; }

		void update_displacement(const double t, Eigen::MatrixXd &sol) const;
		void set_zero(Eigen::MatrixXd &sol) const;

		void clear();

		class Plane;
		inline const std::vector<Plane> &planes() const { return planes_; };

	private:
		void append_mesh(const json &mesh_in, const std::string &root_path);
		void append_plane(const json &plane_in);
		void append_ground(const json &ground_in);

		int dim_;
		Eigen::MatrixXd v_;
		Eigen::VectorXi codim_v_;
		Eigen::MatrixXi f_;
		Eigen::MatrixXi e_;
		Eigen::MatrixXi f_2_e_;

		Eigen::VectorXi in_v_;
		Eigen::MatrixXi in_f_;
		Eigen::MatrixXi in_e_;

		std::vector<std::array<ExpressionValue, 3>> displacements_;
		std::vector<std::shared_ptr<Interpolation>> displacements_interpolation_;

		std::vector<int> endings_;

		std::vector<Plane> planes_;
	};

	class Obstacle::Plane
	{
	public:
		Plane(const VectorNd &origin, const VectorNd &normal)
			: dim_(origin.size()), origin_(origin), normal_(normal)
		{
			assert(origin.size() == normal.size());
			assert(!normal.isZero());
			normal_.normalize();
			construct_vis_mesh();
		}

		inline const VectorNd &origin() const { return origin_; }
		inline const VectorNd &normal() const { return normal_; }

		inline const Eigen::MatrixXd &vis_v() const { return vis_v_; }
		inline const Eigen::MatrixXi &vis_f() const { return vis_f_; }
		inline const Eigen::MatrixXi &vis_e() const { return vis_e_; }

	protected:
		void construct_vis_mesh();

		int dim_;

		VectorNd origin_;
		VectorNd normal_;

		Eigen::MatrixXd vis_v_;
		Eigen::MatrixXi vis_f_;
		Eigen::MatrixXi vis_e_;
	};
} // namespace polyfem
