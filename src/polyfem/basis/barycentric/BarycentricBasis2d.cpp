////////////////////////////////////////////////////////////////////////////////
#include "BarycentricBasis2d.hpp"
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
			std::vector<int> compute_nonzero_bases_ids(const Mesh2D &mesh, const int element_index,
													   const std::vector<ElementBases> &bases,
													   const Eigen::MatrixXd &poly, std::vector<LocalBoundary> &local_boundary)
			{
				const int n_edges = mesh.n_face_vertices(element_index);

				std::vector<int> local_to_global(n_edges);
				LocalBoundary lb(element_index, BoundaryType::POLYGON);

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

		int BarycentricBasis2d::build_bases(
			const std::string &assembler_name,
			const int dim,
			const Mesh2D &mesh,
			const int n_bases,
			const int quadrature_order,
			const int mass_quadrature_order,
			const std::function<void(const Eigen::MatrixXd &, const Eigen::RowVector2d &, Eigen::MatrixXd &, const double)> bc,
			const std::function<void(const Eigen::MatrixXd &, const Eigen::RowVector2d &, Eigen::MatrixXd &, const double)> bc_prime,
			std::vector<ElementBases> &bases,
			std::vector<LocalBoundary> &local_boundary,
			std::map<int, Eigen::MatrixXd> &mapped_boundary)
		{
			assert(!mesh.is_volume());

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

				std::vector<int> local_to_global = compute_nonzero_bases_ids(mesh, e, bases, polygon, local_boundary);

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
				poly_quadr.get_quadrature(polygon, quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler_name, 1, AssemblerUtils::BasisType::POLY, 2), tmp_quadrature);

				Quadrature tmp_mass_quadrature;
				poly_quadr.get_quadrature(polygon, mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", 1, AssemblerUtils::BasisType::POLY, 2), tmp_mass_quadrature);

				b.set_quadrature([tmp_quadrature](Quadrature &quad) { quad = tmp_quadrature; });
				b.set_mass_quadrature([tmp_mass_quadrature](Quadrature &quad) { quad = tmp_mass_quadrature; });

				const double tol = 1e-10;
				b.set_bases_func([polygon, tol, bc](const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &val) {
					Eigen::MatrixXd tmp;
					val.resize(polygon.rows());
					for (size_t i = 0; i < polygon.rows(); ++i)
					{
						val[i].val.resize(uv.rows(), 1);
					}

					for (int i = 0; i < uv.rows(); ++i)
					{
						bc(polygon, uv.row(i), tmp, tol);

						for (size_t j = 0; j < tmp.size(); ++j)
						{
							val[j].val(i) = tmp(j);
						}
					}
				});
				b.set_grads_func([polygon, tol, bc_prime](const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &val) {
					Eigen::MatrixXd tmp;
					val.resize(polygon.rows());
					for (size_t i = 0; i < polygon.rows(); ++i)
					{
						val[i].grad.resize(uv.rows(), 2);
					}

					for (int i = 0; i < uv.rows(); ++i)
					{
						bc_prime(polygon, uv.row(i), tmp, tol);
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
