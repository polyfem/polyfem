////////////////////////////////////////////////////////////////////////////////
#include "MVPolygonalBasis2d.hpp"
#include <polyfem/quadrature/PolygonQuadrature.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <memory>

namespace polyfem
{
	using namespace assembler;
	using namespace mesh;
	using namespace quadrature;

	namespace basis
	{

		namespace
		{

			template <typename Expr>
			inline Eigen::RowVector2d rotatePi_2(const Expr &p) // rotation of pi/2
			{
				return Eigen::RowVector2d(-p(1), p(0));
			}

			std::vector<int> compute_nonzero_bases_ids(const Mesh2D &mesh, const int element_index,
													   const std::vector<ElementBases> &bases,
													   const std::map<int, InterfaceData> &poly_edge_to_data, const Eigen::MatrixXd &poly, std::vector<LocalBoundary> &local_boundary)
			{
				const int n_edges = mesh.n_face_vertices(element_index);

				std::vector<int> local_to_global(n_edges);
				LocalBoundary lb(element_index, BoundaryType::Polygon);

				Navigation::Index index = mesh.get_index_from_face(element_index);
				for (int i = 0; i < n_edges; ++i)
				{
					bool found = false;

					Navigation::Index index1 = mesh.next_around_vertex(index);
					while (index1.face != index.face)
					{
						if (index1.face < 0)
							break;
						if (found)
							break;

						const ElementBases &bs = bases[index1.face];

						for (const auto &b : bs.bases)
						{
							for (const auto &x : b.global())
							{
								const int global_node_id = x.index;
								if ((x.node - poly.row(i)).norm() < 1e-10)
								{
									local_to_global[i] = global_node_id;
									found = true;
									assert(b.global().size() == 1);
									break;
								}
							}
							if (found)
								break;
						}

						index1 = mesh.next_around_vertex(index1);
					}

					index1 = mesh.next_around_vertex(mesh.switch_edge(index));

					while (index1.face != index.face)
					{
						if (index1.face < 0)
							break;
						if (found)
							break;

						const ElementBases &bs = bases[index1.face];

						for (const auto &b : bs.bases)
						{
							for (const auto &x : b.global())
							{
								const int global_node_id = x.index;
								if ((x.node - poly.row(i)).norm() < 1e-10)
								{
									local_to_global[i] = global_node_id;
									found = true;
									assert(b.global().size() == 1);
									break;
								}
							}
							if (found)
								break;
						}

						index1 = mesh.next_around_vertex(index1);
					}

					if (!found)
						local_to_global[i] = -1;

					if (mesh.is_boundary_edge(index.edge) || mesh.get_boundary_id(index.edge) > 0)
						lb.add_boundary_primitive(index.edge, i);

					index = mesh.next_around_face(index);
				}

				if (!lb.empty())
				{
					local_boundary.emplace_back(lb);
				}

				return local_to_global;
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

				//we are on the vertex
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

				//we are on the edge
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

				//we are on the vertex
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
				products(i) = segments.row(i).dot(segments.row(ip1));

				//we are on the edge
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
					assert(false);
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
			const Mesh2D &mesh,
			const int n_bases,
			const int quadrature_order,
			std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const std::map<int, InterfaceData> &poly_edge_to_data,
			std::vector<LocalBoundary> &local_boundary,
			std::map<int, Eigen::MatrixXd> &mapped_boundary)
		{
			assert(!mesh.is_volume());
			if (poly_edge_to_data.empty())
			{
				return 0;
			}

			const int dim = AssemblerUtils::is_tensor(assembler_name) ? 2 : 1;

			Eigen::MatrixXd polygon;

			// int new_nodes = 0;

			std::map<int, int> new_nodes;

			PolygonQuadrature poly_quadr;
			for (int e = 0; e < mesh.n_elements(); ++e)
			{
				if (!mesh.is_polytope(e))
				{
					continue;
				}

				polygon.resize(mesh.n_face_vertices(e), 2);

				for (int i = 0; i < mesh.n_face_vertices(e); ++i)
				{
					const int gid = mesh.face_vertex(e, i);
					polygon.row(i) = mesh.point(gid);
				}

				std::vector<int> local_to_global = compute_nonzero_bases_ids(mesh, e, bases, poly_edge_to_data, polygon, local_boundary);

				for (int i = 0; i < local_to_global.size(); ++i)
				{
					if (local_to_global[i] >= 0)
						continue;

					const int gid = mesh.face_vertex(e, i);
					const auto other_gid = new_nodes.find(gid);
					if (other_gid != new_nodes.end())
						local_to_global[i] = other_gid->second;
					else
					{
						const int tmp = new_nodes.size() + n_bases;
						new_nodes[gid] = tmp;
						local_to_global[i] = tmp;
					}
				}

				ElementBases &b = bases[e];
				b.has_parameterization = false;

				// Compute quadrature points for the polygon
				Quadrature tmp_quadrature;
				poly_quadr.get_quadrature(polygon, quadrature_order, tmp_quadrature);

				b.set_quadrature([tmp_quadrature](Quadrature &quad) { quad = tmp_quadrature; });

				const double tol = 1e-10;
				b.set_bases_func([polygon, tol](const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &val) {
					Eigen::MatrixXd tmp;
					val.resize(polygon.rows());
					for (size_t i = 0; i < polygon.rows(); ++i)
					{
						val[i].val.resize(uv.rows(), 1);
					}

					for (int i = 0; i < uv.rows(); ++i)
					{
						meanvalue(polygon, uv.row(i), tmp, tol);

						for (size_t j = 0; j < tmp.size(); ++j)
						{
							val[j].val(i) = tmp(j);
						}
					}
				});
				b.set_grads_func([polygon, tol](const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &val) {
					Eigen::MatrixXd tmp;
					val.resize(polygon.rows());
					for (size_t i = 0; i < polygon.rows(); ++i)
					{
						val[i].grad.resize(uv.rows(), 2);
					}

					for (int i = 0; i < uv.rows(); ++i)
					{
						meanvalue_derivative(polygon, uv.row(i), tmp, tol);
						assert(tmp.rows() == polygon.rows());

						for (size_t j = 0; j < tmp.rows(); ++j)
						{
							val[j].grad.row(i) = tmp.row(j);
						}
					}
				});

				b.set_local_node_from_primitive_func([e](const int primitive_id, const Mesh &mesh) {
					const auto &mesh2d = dynamic_cast<const Mesh2D &>(mesh);
					auto index = mesh2d.get_index_from_face(e);

					int le;
					for (le = 0; le < mesh2d.n_face_vertices(e); ++le)
					{
						if (index.edge == primitive_id)
							break;
						index = mesh2d.next_around_face(index);
					}
					assert(index.edge == primitive_id);
					Eigen::VectorXi result(2);
					result(0) = le;
					result(1) = (le + 1) % mesh2d.n_face_vertices(e);
					return result;
				});

				// Set the bases which are nonzero inside the polygon
				const int n_poly_bases = int(local_to_global.size());
				b.bases.resize(n_poly_bases);
				for (int i = 0; i < n_poly_bases; ++i)
				{
					b.bases[i].init(-1, local_to_global[i], i, polygon.row(i));
				}

				// Polygon boundary after geometric mapping from neighboring elements
				mapped_boundary[e] = polygon;
			}

			return new_nodes.size();
		}

	} // namespace basis
} // namespace polyfem
