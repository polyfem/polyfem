#pragma once

#include <polyfem/basis/ElementBases.hpp>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>
#include <spdlog/fmt/fmt.h>

namespace polyfem::utils
{
	class Tree
	{
	public:
		Tree() {}
		Tree(const Tree &T)
		{
			if (T.has_children())
			{
				for (int i = 0; i < T.n_children(); i++)
					this->children.push_back(std::make_unique<Tree>(T.child(i)));
			}
		}
		Tree operator=(const Tree &T)
		{
			if (this != &T)
			{
				this->children.clear();
				if (T.has_children())
				{
					for (int i = 0; i < T.n_children(); i++)
						this->children.push_back(std::make_unique<Tree>(T.child(i)));
				}
			}
			return *this;
		}
		bool merge(const Tree &T, int max_depth = 2)
		{
			bool flag = false;
			if (!T.has_children() || max_depth <= 0)
				return flag;
			if (!this->has_children())
			{
				this->add_children(T.n_children());
				flag = true;
				max_depth--;
			}
			for (int i = 0; i < T.n_children(); i++)
				flag = this->child(i).merge(T.child(i), max_depth) || flag;
			return flag;
		}

		// Debug print
		friend std::ostream &operator<<(
			std::ostream &ost, const Tree &T)
		{
			ost << "(";
			if (T.has_children())
			{
				for (int i = 0; i < T.n_children(); i++)
					ost << T.child(i) << ", ";
			}
			ost << ")";
			return ost;
		}

		bool has_children() const { return !children.empty(); }
		int n_children() const { return children.size(); }
		int depth() const
		{
			if (!has_children())
				return 0;
			int d = 0;
			for (int i = 0; i < n_children(); i++)
				d = std::max(d, child(i).depth());
			return d + 1;
		}
		int n_leaves() const
		{
			if (!has_children())
				return 1;
			int n = 0;
			for (int i = 0; i < n_children(); i++)
				n += child(i).n_leaves();
			return n;
		}
		Tree &child(int i) { return *children[i]; }
		const Tree &child(int i) const { return *children[i]; }
		void add_children(int n)
		{
			for (int i = 0; i < n; i++)
				children.push_back(std::make_unique<Tree>());
		}

	private:
		std::vector<std::unique_ptr<Tree>> children;
	};

	/// @brief Vertices of child @p i of the reference simplex under the
	/// regular 1:2^dim split used for quadrature refinement, in the
	/// parent's reference coordinates. Rows are vertices.
	Eigen::MatrixXd refined_nodes(const int dim, const int i);

	/// @brief Vertices of every leaf simplex of @p tree, obtained by
	/// recursively applying refined_nodes() to the simplex @p pts.
	/// @param pts vertices of the root simplex, one per row
	/// @param tree refinement hierarchy
	/// @return stacked leaf vertices ((dim+1) rows per leaf) and the depth of each leaf
	std::tuple<Eigen::MatrixXd, std::vector<int>> extract_subelement(const Eigen::MatrixXd &pts, const Tree &tree);

	/// @brief One element's control points, as the conservative checks see
	/// them: the nodes of order max(displacement, geometry).
	struct ElementControlPoints
	{
		int order = 0;
		/// Rest positions, one row per control point (same values as the
		/// element's rows of rest_nodes()).
		Eigen::MatrixXd rest;
		/// Global node whose displacement each control point copies, when the
		/// displacement basis is nodal there (as in displacement_nodes());
		/// empty when the displacement has to be evaluated instead.
		std::vector<int> nodes;
	};

	ElementControlPoints element_control_points(const int dim, const basis::ElementBases &basis, const basis::ElementBases &gbasis);

	/// @brief Control points of every element, to compute once and reuse.
	std::vector<ElementControlPoints> element_control_points(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases);

	/// @brief Displacement of one element's control points: the same numbers
	/// displacement_nodes() gives the checks.
	Eigen::MatrixXd control_point_displacement(const int dim, const ElementControlPoints &cp, const basis::ElementBases &basis, const basis::ElementBases &gbasis, const Eigen::Ref<const Eigen::VectorXd> &u);

	/// @brief Jacobian determinant of one element's map at reference points
	/// @p uv, for the elastic energy under a conservative check.
	///
	/// It is evaluated with the generated expression the checks use (JacEval),
	/// in interval arithmetic, on the same inputs: the rest positions and the
	/// displacements of the control points, separately. The interval therefore
	/// provably contains the true value, and if the check accepted this
	/// configuration the true value is positive, so the part of the interval
	/// below zero is discarded: each entry is the midpoint of [max(lo, 0), hi],
	/// positive whenever hi > 0 (log J stays finite wherever the check passed),
	/// and NaN when hi <= 0 (certainly inverted there).
	/// @param rest Rest positions of the control points (ElementControlPoints::rest).
	/// @param disp Their displacements (control_point_displacement()).
	Eigen::VectorXd robust_evaluate_jacobian(
		const int order,
		const Eigen::MatrixXd &rest,
		const Eigen::MatrixXd &disp,
		const Eigen::MatrixXd &uv);

	/// @param rest Rest positions from rest_nodes(); computed here if null.
	std::vector<int> count_invalid(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u,
		const unsigned max_iter = 1'000'000,
		const Eigen::MatrixXd *rest = nullptr);

	/// @brief Certify that no element of configuration @p u is inverted.
	/// @return (valid, id of the first invalid element or -1, witness tree).
	///         The tree is the subdivision path to where the violation was
	///         found inside that element, in the same child numbering as the
	///         quadrature refinement; it is empty for P1 (constant Jacobian,
	///         nothing to localize), when the violation is already visible at
	///         the element's corners, or when the solver gave up.
	/// @param rest Rest positions from rest_nodes(); computed here if null.
	std::tuple<bool, int, Tree> is_valid(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u,
		const double threshold = 0,
		const unsigned max_iter = 1'000'000,
		const Eigen::MatrixXd *rest = nullptr);

	/// @brief Rest positions of every element's control points: the nodes of
	/// order max(displacement, geometry), mapped by the geometry. Compute it
	/// once (in round-to-nearest) and pass it to the checks below, so they all
	/// see the same numbers, which are also the ones the energy sees through
	/// element_control_points().
	Eigen::MatrixXd rest_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases);

	/// @brief Displacement of every element's control points. When the
	/// displacement basis is nodal at those points (conforming, of the
	/// control-point order) each is an entry of @p u, copied; otherwise it is
	/// evaluated there, which rounds. @p exact reports which.
	Eigen::MatrixXd displacement_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const Eigen::VectorXd &u, bool *exact = nullptr);

	/// @brief [lo, hi] with each entry of @p x widened by @p n representable
	/// values on each side (hocgv-miso fix.pdf, steps 2-3).
	std::pair<Eigen::MatrixXd, Eigen::MatrixXd> fatten(const Eigen::MatrixXd &x, const int n);

	/// @brief Largest fraction of the step u1 -> u2 over which no element
	/// inverts. The displacements are widened as in hocgv-miso fix.pdf (u1 by
	/// one representable value, u2 by two), so the certificate also covers the
	/// correctly rounded configuration the solver takes after capping the
	/// step, not just the exact segment.
	/// @param rest Rest positions from rest_nodes(); computed here if null.
	std::tuple<double, int, double, Tree> max_time_step(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u1,
		const Eigen::VectorXd &u2,
		double precision = .25,
		double threshold = 0,
		const unsigned max_iter = 1'000'000,
		const Eigen::MatrixXd *rest = nullptr);

	Eigen::MatrixXd extract_nodes(const int dim, const basis::ElementBases &basis, const basis::ElementBases &gbasis, const Eigen::VectorXd &u, int order);
	Eigen::MatrixXd extract_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const Eigen::VectorXd &u, int order, int n_elem = -1);
} // namespace polyfem::utils

template <>
struct fmt::formatter<polyfem::utils::Tree> : fmt::formatter<fmt::string_view>
{
	format_context::iterator format(const polyfem::utils::Tree &tree, fmt::format_context &ctx) const
	{
		std::ostringstream oss;
		oss << tree;
		return fmt::formatter<fmt::string_view>::format(oss.str(), ctx);
	}
};
