////////////////////////////////////////////////////////////////////////////////
#include "MVPolygonalBasis2d.hpp"
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

		void MVPolygonalBasis2d::meanvalue(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &b, const double tol)
		{
			const int n_boundary = polygon.rows();

			Eigen::MatrixXd segments(n_boundary, 2);
			Eigen::VectorXd radii(n_boundary);
			Eigen::VectorXd areas(n_boundary);
			Eigen::VectorXd products(n_boundary);
			Eigen::VectorXd tangents(n_boundary);
			Eigen::Matrix2d mat;

			b.resize(n_boundary, 1);
			b.setZero();

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
				products(i) = segments.row(i).dot(segments.row(ip1));

				// we are on the edge
				if (fabs(areas[i]) < tol && products(i) < 0)
				{
					const double denominator = 1.0 / (radii(i) + radii(ip1));

					b(i) = radii(ip1) * denominator;
					b(ip1) = radii(i) * denominator;

					return;
				}
			}

			for (int i = 0; i < n_boundary; ++i)
			{
				const int ip1 = (i + 1) == n_boundary ? 0 : (i + 1);

				tangents(i) = areas(i) / (radii(i) * radii(ip1) + products(i));
			}

			double W = 0;
			for (int i = 0; i < n_boundary; ++i)
			{
				const int im1 = i == 0 ? (n_boundary - 1) : (i - 1);

				b(i) = (tangents(im1) + tangents(i)) / radii(i);
				W += b(i);
			}

			b /= W;
		}

		void MVPolygonalBasis2d::meanvalue_derivative(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &derivatives, const double tol)
		{
			const int n_boundary = polygon.rows();

			// b.resize(n_boundary*n_points);
			// std::fill(b.begin(), b.end(), 0);

			derivatives.resize(n_boundary, 2);
			derivatives.setZero();

			Eigen::MatrixXd segments(n_boundary, 2);
			Eigen::VectorXd radii(n_boundary);
			Eigen::VectorXd areas(n_boundary);
			Eigen::VectorXd products(n_boundary);
			Eigen::VectorXd tangents(n_boundary);
			Eigen::Matrix2d mat;

			Eigen::MatrixXd areas_prime(n_boundary, 2);
			Eigen::MatrixXd products_prime(n_boundary, 2);
			Eigen::MatrixXd radii_prime(n_boundary, 2);
			Eigen::MatrixXd tangents_prime(n_boundary, 2);
			Eigen::MatrixXd w_prime(n_boundary, 2);

			// Eigen::MatrixXd b(n_boundary, 1);

			for (int i = 0; i < n_boundary; ++i)
			{
				segments.row(i) = polygon.row(i) - point;

				radii(i) = segments.row(i).norm();

				// we are on the vertex
				if (radii(i) < tol)
				{
					// assert(false);
					//  b(i) = 1;
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
				products(i) = segments.row(i).dot(segments.row(ip1));

				// we are on the edge
				if (fabs(areas[i]) < tol && products(i) < 0)
				{
					// const double denominator = 1.0/(radii(i) + radii(ip1));
					// w0 = radii(ip1); // * denominator;
					// w1 = radii(i); // * denominator;

					// //https://link.springer.com/article/10.1007/s00371-013-0889-y
					//             const Eigen::RowVector2d NE = rotatePi_2(polygon.row(ip1) - polygon.row(i));
					//             const double sqrlengthE = NE.squaredNorm();

					//             const Eigen::RowVector2d N0 = rotatePi_2(point - polygon.row(i));
					//             const Eigen::RowVector2d N1 = rotatePi_2( polygon.row(ip1) - point);
					//             const double N0norm = N0.norm();
					//             const double N1norm = N1.norm();

					//             const Eigen::RowVector2d gradw0 = (N0.dot(N1) / (2*N0norm*N0norm*N0norm) + 1./(2.*N1norm) + 1./N0norm - 1./N1norm ) * NE / sqrlengthE;
					//             const Eigen::RowVector2d gradw1 = (1./(2*N1norm) + N0.dot(N1)/(2*N1norm*N1norm*N1norm) - 1./N0norm + 1./N1norm ) * NE / sqrlengthE;

					//             w_prime.setZero();
					//             w_prime.row(i) = gradw0;
					//             w_prime.row(ip1) = gradw1;

					//             assert(on_edge == -1);
					//             on_edge = i;
					// continue;

					// w_gradients_on_edges[e] = std::pair<point_t,point_t>(gradw0,gradw1);

					// w_gradients[e0] += gradw0;
					// w_gradients[e1] += gradw1;
					// assert(false);
					return;
				}
				const Eigen::RowVector2d vi = polygon.row(i);
				const Eigen::RowVector2d vip1 = polygon.row(ip1);

				areas_prime(i, 0) = vi(1) - vip1(1);
				areas_prime(i, 1) = vip1(0) - vi(0);

				products_prime.row(i) = 2 * point - vi - vip1;
				radii_prime.row(i) = (point - vi) / radii(i);
			}

			for (int i = 0; i < n_boundary; ++i)
			{
				// if(i == on_edge)
				// 	continue;

				const int ip1 = (i + 1) == n_boundary ? 0 : (i + 1);

				const double denominator = radii(i) * radii(ip1) + products(i);
				const Eigen::RowVector2d denominator_prime = radii_prime.row(i) * radii(ip1) + radii(i) * radii_prime.row(ip1) + products_prime.row(i);

				tangents_prime.row(i) = (areas_prime.row(i) * denominator - areas(i) * denominator_prime) / (denominator * denominator);
				tangents(i) = areas(i) / denominator;
			}

			double W = 0;
			Eigen::RowVector2d W_prime;
			W_prime.setZero();

			for (int i = 0; i < n_boundary; ++i)
			{
				const int im1 = (i > 0) ? (i - 1) : (n_boundary - 1);

				if (i != on_edge && im1 != on_edge)
					w_prime.row(i) = ((tangents_prime.row(im1) + tangents_prime.row(i)) * radii(i) - (tangents(im1) + tangents(i)) * radii_prime.row(i)) / (radii(i) * radii(i));
				;

				W_prime += w_prime.row(i);
				if (i == on_edge)
					W += w0;
				else if (im1 == on_edge)
					W += w1;
				else if (on_edge < 0)
					W += (tangents(im1) + tangents(i)) / radii(i);
			}

			for (int i = 0; i < n_boundary; ++i)
			{
				const int im1 = (i > 0) ? (i - 1) : (n_boundary - 1);

				double wi;
				if (i == on_edge)
					wi = w0;
				else if (im1 == on_edge)
					wi = w1;
				else if (on_edge < 0)
					wi = (tangents(im1) + tangents(i)) / radii(i);

				derivatives.row(i) = (w_prime.row(i) * W - wi * W_prime) / (W * W);
			}
		}

		int MVPolygonalBasis2d::build_bases(
			const std::string &assembler_name,
			const int dim,
			const mesh::Mesh2D &mesh,
			const int n_bases,
			const int quadrature_order,
			const int mass_quadrature_order,
			std::vector<ElementBases> &bases,
			std::vector<mesh::LocalBoundary> &local_boundary,
			std::map<int, Eigen::MatrixXd> &mapped_boundary)
		{
			return BarycentricBasis2d::build_bases(assembler_name, dim, mesh, n_bases, quadrature_order, mass_quadrature_order, meanvalue, meanvalue_derivative, bases, local_boundary, mapped_boundary);
		}

	} // namespace basis
} // namespace polyfem
