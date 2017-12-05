#ifndef TP_B_SPLINE_HPP
#define TP_B_SPLINE_HPP



#include <vector>
#include <cassert>

#include <Eigen/Dense>

namespace poly_fem {
	class TensorIndex
	{
	public:
		TensorIndex();
		TensorIndex(const int dim1, const int dim2, const int dim3);

		void init(const int dim1, const int dim2, const int dim3);

		int index_for(const int i, const int j, const int k) const;
		int size() const;
		int operator[](const int i) const;

	private:
		int dims_[3];
	};



	class TensorProductBSpline {
	private:
		int degree_u_, degree_v_;
		TensorIndex tensor_index_;

		std::vector<double> knots_u_, knots_v_;
		std::vector<double> control_points_;

	public:
		inline int degree_u() const { return degree_u_; }
		inline int degree_v() const { return degree_v_; }
		inline int dim() const { return tensor_index_[2]; }

		inline const std::vector<double> &controlPoints() const { return control_points_; }

		void init(const std::vector<double> &knots_u, const std::vector<double> &knots_v, const std::vector<double> &control_points, const int n_control_u, const int n_control_v, const int dim = 2);
		void init(const int degree_u, const int degree_v, const std::vector<double> &knots_u, const std::vector<double> &knots_v, const std::vector<double> &control_points, const int n_control_u, const int n_control_v, const int dim = 2);

		void interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
		void interpolate(const double u, const double v, std::vector<double> &result) const;

		void derivative(TensorProductBSpline &dx, TensorProductBSpline &dy) const;

	private:
		inline double ctrl_pts_at(const int i, const int j, const int k) const
		{
			return control_points_[tensor_index_.index_for(i,j,k)];
		}
	};
}
#endif //TP_B_SPLINE_HPP
