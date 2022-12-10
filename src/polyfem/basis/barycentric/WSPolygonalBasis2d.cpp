////////////////////////////////////////////////////////////////////////////////
#include "WSPolygonalBasis2d.hpp"
#include "BarycentricBasis2d.hpp"

#include <memory>

namespace polyfem
{
	namespace basis
	{

		namespace
		{
			template <typename Expr>
			inline Eigen::RowVector2d rotatePi_2(const Expr &p) // rotation of pi/2
			{
				return Eigen::RowVector2d(-p(1), p(0));
			}
		} // anonymous namespace

		////////////////////////////////////////////////////////////////////////////////

		void WSPolygonalBasis2d::wachspress(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &b, const double tol)
		{
			const int n_boundary = polygon.rows();

			Eigen::MatrixXd segments(n_boundary, 2);
			Eigen::VectorXd radii(n_boundary);
			Eigen::VectorXd areas(n_boundary);
			Eigen::VectorXd Cs(n_boundary);
			Eigen::Matrix2d mat;

			b.resize(n_boundary, 1);
			b.setZero();

			for (int i = 0; i < n_boundary; ++i)
			{
				const int ip1 = (i + 1) == n_boundary ? 0 : (i + 1);
				const int im1 = i == 0 ? (n_boundary - 1) : (i - 1);

				mat.row(0) = polygon.row(im1) - polygon.row(i);
				mat.row(1) = polygon.row(ip1) - polygon.row(i);

				Cs(i) = mat.determinant();
			}

			for (int i = 0; i < n_boundary; ++i)
			{
				segments.row(i) = polygon.row(i) - point;
				radii(i) = segments.row(i).norm();

				// we are on the vertex
				if (radii(i) < tol)
				{
					b(i) = 1;
					return;
				}
			}

			for (int i = 0; i < n_boundary; ++i)
			{
				const int ip1 = (i + 1) == n_boundary ? 0 : (i + 1);

				mat.row(0) = segments.row(i);
				mat.row(1) = segments.row(ip1);

				areas(i) = mat.determinant();
				const double prod = segments.row(i).dot(segments.row(ip1));

				// we are on the edge
				if (fabs(areas[i]) < tol && prod < 0)
				{
					const double denominator = 1.0 / (radii(i) + radii(ip1));

					b(i) = radii(ip1) * denominator;
					b(ip1) = radii(i) * denominator;

					return;
				}
			}

			double W = 0;
			for (int i = 0; i < n_boundary; ++i)
			{
				const int im1 = i == 0 ? (n_boundary - 1) : (i - 1);

				b(i) = Cs(i) / (areas(im1) * areas(i));
				W += b(i);
			}

			b /= W;
		}

		void WSPolygonalBasis2d::wachspress_derivative(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &derivatives, const double tol)
		{
			const int n_boundary = polygon.rows();

			derivatives.resize(n_boundary, 2);
			derivatives.setZero();

			Eigen::MatrixXd segments(n_boundary, 2);
			Eigen::VectorXd radii(n_boundary);
			Eigen::VectorXd areas(n_boundary);
			Eigen::VectorXd Cs(n_boundary);
			Eigen::Matrix2d mat;

			Eigen::MatrixXd areas_prime(n_boundary, 2);
			Eigen::MatrixXd w_prime(n_boundary, 2);

			for (int i = 0; i < n_boundary; ++i)
			{
				const int ip1 = (i + 1) == n_boundary ? 0 : (i + 1);
				const int im1 = i == 0 ? (n_boundary - 1) : (i - 1);

				mat.row(0) = polygon.row(im1) - polygon.row(i);
				mat.row(1) = polygon.row(ip1) - polygon.row(i);

				Cs(i) = mat.determinant();
			}

			for (int i = 0; i < n_boundary; ++i)
			{
				segments.row(i) = polygon.row(i) - point;
				radii(i) = segments.row(i).norm();

				// we are on the vertex
				if (radii(i) < tol)
				{
					assert(false);
					// b(i) = 1;
					return;
				}
			}

			int on_edge = -1;
			double w0 = 0, w1 = 0;

			for (int i = 0; i < n_boundary; ++i)
			{
				const int ip1 = (i + 1) == n_boundary ? 0 : (i + 1);

				mat.row(0) = segments.row(i);
				mat.row(1) = segments.row(ip1);

				areas(i) = mat.determinant();
				const double prod = segments.row(i).dot(segments.row(ip1));

				// we are on the edge
				if (fabs(areas[i]) < tol && prod < 0)
				{
					assert(false);
					return;
				}

				const Eigen::RowVector2d vi = polygon.row(i);
				const Eigen::RowVector2d vip1 = polygon.row(ip1);

				areas_prime(i, 0) = vi(1) - vip1(1);
				areas_prime(i, 1) = vip1(0) - vi(0);
			}

			double W = 0;
			Eigen::RowVector2d W_prime;
			W_prime.setZero();

			for (int i = 0; i < n_boundary; ++i)
			{
				const int im1 = (i > 0) ? (i - 1) : (n_boundary - 1);

				w_prime.row(i) = Cs(i) / (areas(im1) * areas(i)) / (areas(im1) * areas(i)) * (areas_prime.row(im1) * areas(i) + areas(im1) * areas_prime.row(i));

				W_prime += w_prime.row(i);

				W += Cs(i) / (areas(im1) * areas(i));
			}

			for (int i = 0; i < n_boundary; ++i)
			{
				const int im1 = (i > 0) ? (i - 1) : (n_boundary - 1);

				const double wi = Cs(i) / (areas(im1) * areas(i));
				derivatives.row(i) = -(w_prime.row(i) * W - wi * W_prime) / (W * W);
			}
		}

		int WSPolygonalBasis2d::build_bases(
			const std::string &assembler_name,
			const mesh::Mesh2D &mesh,
			const int n_bases,
			const int quadrature_order,
			const int mass_quadrature_order,
			std::vector<ElementBases> &bases,
			std::vector<mesh::LocalBoundary> &local_boundary,
			std::map<int, Eigen::MatrixXd> &mapped_boundary)
		{
			return BarycentricBasis2d::build_bases(assembler_name, mesh, n_bases, quadrature_order, mass_quadrature_order, wachspress, wachspress_derivative, bases, local_boundary, mapped_boundary);
		}

	} // namespace basis
} // namespace polyfem
