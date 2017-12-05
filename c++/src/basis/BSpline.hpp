#ifndef B_SPLINE_HPP
#define B_SPLINE_HPP

#include <cassert>
#include <vector>

namespace poly_fem {
	class BSpline
	{
	private:
		int degree_;
		int dim_;
		std::vector<double> knots_;
		std::vector<double> control_points_;

	public:
		inline int degree() const { return degree_; }
		inline int dim() const { return dim_; }

		inline const std::vector<double> &control_points() const { return control_points_; }

		void init(const std::vector<double> &knots, const std::vector<double> &control_points, int dim = 2);
		void init(const int degree, const std::vector<double> &knots, const std::vector<double> &control_points, int dim = 2);

		void interpolate(const std::vector<double> &ts, std::vector<double> &result) const;
		void interpolate(const double t, std::vector<double> &result) const;

		void derivative(BSpline &result) const;

	private:
		int find_interval(const double t) const;
		int first_interval(const int interval, const int degree) const;

		void find_edges(const int first_interval, const int degree, std::vector<double> &edges) const;
		void create_new_edges(const std::vector<double> &weigths, const std::vector<double> &edges, std::vector<double> &new_edges) const;

		void compute_weigth(const double t, const int first_interval, const int degree, std::vector<double> &weigths) const;
	};
}
#endif //B_SPLINE_HPP
