
#include "APriori.hpp"

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/utils/Logger.hpp>

namespace polyfem::refinement
{
	using namespace mesh;
	using namespace utils;

	namespace
	{
		template <typename V1, typename V2>
		double angle2(const V1 &v1, const V2 &v2)
		{
			assert(v1.size() == 2);
			assert(v2.size() == 2);
			return std::abs(atan2(v1(0) * v2(1) - v1(1) * v2(0), v1.dot(v2)));
		}

		template <typename V1, typename V2>
		double angle3(const V1 &v1, const V2 &v2)
		{
			assert(v1.size() == 3);
			assert(v2.size() == 3);
			return std::abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));
		}

	} // namespace

	double get_opt_p(bool h1_formula, double B,
					 double h_ref, int p_ref, double rho_ref,
					 double h, double rho, int p_max)
	{
		const double sigma_ref = rho_ref / h_ref;
		const double sigma = rho / h;

		const double ptmp = h1_formula ? (std::log(B * std::pow(h_ref, p_ref + 1) * rho / (h * rho_ref)) / std::log(h)) : (std::log(B * std::pow(h_ref, p_ref + 1) * sigma * sigma / sigma_ref / sigma_ref) - std::log(h)) / std::log(h);
		// (std::log(B*std::pow(h_ref, p_ref + 2)*rho*rho / (h * h *rho_ref*rho_ref))/std::log(h));

		return std::min(std::max(p_ref, (int)std::round(ptmp)), p_max);
	}

	void APriori::p_refine(
		const Mesh2D &mesh2d,
		const double B,
		const bool h1_formula,
		const int base_p,
		const int discr_order_max,
		io::OutStatsData &stats,
		Eigen::VectorXi &disc_orders)
	{
		stats.max_angle = 0;
		// static const int max_angles = 5;
		// static const double angles[max_angles] = {0, 170./180.*M_PI, 179./180.*M_PI, 179.9/180.* M_PI, M_PI};

		Eigen::MatrixXd p0, p1;
		mesh2d.get_edges(p0, p1);
		const auto tmp = p0 - p1;
		const double h_ref = tmp.rowwise().norm().mean();

		const double rho_ref = sqrt(3.0) / 6.0 * h_ref;
		const int p_max = std::min(autogen::MAX_P_BASES, discr_order_max);

		stats.sigma_avg = 0;
		stats.sigma_max = 0;
		stats.sigma_min = std::numeric_limits<double>::max();

		for (int f = 0; f < mesh2d.n_faces(); ++f)
		{
			if (!mesh2d.is_simplex(f))
				continue;

			auto v0 = mesh2d.point(mesh2d.face_vertex(f, 0));
			auto v1 = mesh2d.point(mesh2d.face_vertex(f, 1));
			auto v2 = mesh2d.point(mesh2d.face_vertex(f, 2));

			const RowVectorNd e0 = v1 - v0;
			const RowVectorNd e1 = v2 - v1;
			const RowVectorNd e2 = v0 - v2;

			const double e0n = e0.norm();
			const double e1n = e1.norm();
			const double e2n = e2.norm();

			const double alpha0 = angle2(e0, -e2);
			const double alpha1 = angle2(e1, -e0);
			const double alpha2 = angle2(e2, -e1);

			const double P = e0n + e1n + e2n;
			const double A = std::abs(e1(0) * e2(1) - e1(1) * e2(0)) / 2;
			const double rho = 2 * A / P;
			const double hp = std::max(e0n, std::max(e1n, e2n));
			const double sigma = rho / hp;

			stats.sigma_avg += sigma;
			stats.sigma_max = std::max(stats.sigma_max, sigma);
			stats.sigma_min = std::min(stats.sigma_min, sigma);

			const int p = get_opt_p(h1_formula, B, h_ref, base_p, rho_ref, hp, rho, p_max);

			if (p > disc_orders[f])
				disc_orders[f] = p;
			auto index = mesh2d.get_index_from_face(f);

			for (int lv = 0; lv < 3; ++lv)
			{
				auto nav = mesh2d.switch_face(index);

				if (nav.face >= 0)
				{
					if (p > disc_orders[nav.face])
						disc_orders[nav.face] = p;
				}

				index = mesh2d.next_around_face(index);
			}

			stats.max_angle = std::max(stats.max_angle, alpha0);
			stats.max_angle = std::max(stats.max_angle, alpha1);
			stats.max_angle = std::max(stats.max_angle, alpha2);
		}

		stats.sigma_avg /= mesh2d.n_faces();
		stats.max_angle = stats.max_angle / M_PI * 180.;
		logger().info("using B={} with {} estimate max_angle {}", B, (h1_formula ? "H1" : "L2"), stats.max_angle);
		logger().info("average sigma: {}", stats.sigma_avg);
		logger().info("min sigma: {}", stats.sigma_min);
		logger().info("max sigma: {}", stats.sigma_max);

		logger().info("num_p1 {}", (disc_orders.array() == 1).count());
		logger().info("num_p2 {}", (disc_orders.array() == 2).count());
		logger().info("num_p3 {}", (disc_orders.array() == 3).count());
		logger().info("num_p4 {}", (disc_orders.array() == 4).count());
		logger().info("num_p5 {}", (disc_orders.array() == 5).count());
	}

	void APriori::p_refine(
		const Mesh3D &mesh3d,
		const double B,
		const bool h1_formula,
		const int base_p,
		const int discr_order_max,
		io::OutStatsData &stats,
		Eigen::VectorXi &disc_orders)
	{
		stats.max_angle = 0;

		Eigen::MatrixXd p0, p1;
		mesh3d.get_edges(p0, p1);
		const auto tmp = p0 - p1;
		const double h_ref = tmp.rowwise().norm().mean();
		const double rho_ref = sqrt(6.) / 12. * h_ref;
		const int p_max = std::min(autogen::MAX_P_BASES, discr_order_max);

		stats.sigma_avg = 0;
		stats.sigma_max = 0;
		stats.sigma_min = std::numeric_limits<double>::max();

		for (int c = 0; c < mesh3d.n_cells(); ++c)
		{
			if (!mesh3d.is_simplex(c))
				continue;

			const auto v0 = mesh3d.point(mesh3d.cell_vertex(c, 0));
			const auto v1 = mesh3d.point(mesh3d.cell_vertex(c, 1));
			const auto v2 = mesh3d.point(mesh3d.cell_vertex(c, 2));
			const auto v3 = mesh3d.point(mesh3d.cell_vertex(c, 3));

			Eigen::Matrix<double, 6, 3> e;
			e.row(0) = v0 - v1;
			e.row(1) = v1 - v2;
			e.row(2) = v2 - v0;

			e.row(3) = v0 - v3;
			e.row(4) = v1 - v3;
			e.row(5) = v2 - v3;

			Eigen::Matrix<double, 6, 1> en = e.rowwise().norm();

			Eigen::Matrix<double, 3 * 4, 1> alpha;
			alpha(0) = angle3(e.row(0), -e.row(1));
			alpha(1) = angle3(e.row(1), -e.row(2));
			alpha(2) = angle3(e.row(2), -e.row(0));
			alpha(3) = angle3(e.row(0), -e.row(4));
			alpha(4) = angle3(e.row(4), e.row(3));
			alpha(5) = angle3(-e.row(3), -e.row(0));
			alpha(6) = angle3(-e.row(4), -e.row(1));
			alpha(7) = angle3(e.row(1), -e.row(5));
			alpha(8) = angle3(e.row(5), e.row(4));
			alpha(9) = angle3(-e.row(2), -e.row(5));
			alpha(10) = angle3(e.row(5), e.row(3));
			alpha(11) = angle3(-e.row(3), e.row(2));

			const double S = (e.row(0).cross(e.row(1)).norm() + e.row(0).cross(e.row(4)).norm() + e.row(4).cross(e.row(1)).norm() + e.row(2).cross(e.row(5)).norm()) / 2;
			const double V = std::abs(e.row(3).dot(e.row(2).cross(-e.row(0)))) / 6;
			const double rho = 3 * V / S;
			const double hp = en.maxCoeff();

			stats.sigma_avg += rho / hp;
			stats.sigma_max = std::max(stats.sigma_max, rho / hp);
			stats.sigma_min = std::min(stats.sigma_min, rho / hp);

			const int p = get_opt_p(h1_formula, B, h_ref, base_p, rho_ref, hp, rho, p_max);

			if (p > disc_orders[c])
				disc_orders[c] = p;

			for (int le = 0; le < 6; ++le)
			{
				const int e_id = mesh3d.cell_edge(c, le);
				const auto cells = mesh3d.edge_neighs(e_id);

				for (auto c_id : cells)
				{
					if (p > disc_orders[c_id])
						disc_orders[c_id] = p;
				}
			}

			stats.max_angle = std::max(stats.max_angle, alpha.maxCoeff());
		}

		stats.max_angle = stats.max_angle / M_PI * 180.;
		stats.sigma_avg /= mesh3d.n_elements();

		logger().info("using B={} with {} estimate max_angle {}", B, (h1_formula ? "H1" : "L2"), stats.max_angle);
		logger().info("average sigma: {}", stats.sigma_avg);
		logger().info("min sigma: {}", stats.sigma_min);
		logger().info("max sigma: {}", stats.sigma_max);

		logger().info("num_p1 {}", (disc_orders.array() == 1).count());
		logger().info("num_p2 {}", (disc_orders.array() == 2).count());
		logger().info("num_p3 {}", (disc_orders.array() == 3).count());
		logger().info("num_p4 {}", (disc_orders.array() == 4).count());
		logger().info("num_p5 {}", (disc_orders.array() == 5).count());
	}

	void APriori::p_refine(
		const mesh::Mesh &mesh,
		const double B,
		const bool h1_formula,
		const int base_p,
		const int discr_order_max,
		io::OutStatsData &stats,
		Eigen::VectorXi &disc_orders)
	{
		if (mesh.is_volume())
			p_refine(static_cast<const Mesh3D &>(mesh), B, h1_formula, base_p, discr_order_max, stats, disc_orders);
		else
			p_refine(static_cast<const Mesh2D &>(mesh), B, h1_formula, base_p, discr_order_max, stats, disc_orders);
	}
} // namespace polyfem::refinement
