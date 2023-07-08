#include "BoundarySampler.hpp"

#include <polyfem/quadrature/LineQuadrature.hpp>
#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/QuadQuadrature.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>

#include <cassert>

namespace polyfem
{
	using namespace mesh;
	using namespace quadrature;
	namespace utils
	{
		namespace
		{
			Eigen::RowVector2d quad_local_node_coordinates(int local_index)
			{
				assert(local_index >= 0 && local_index < 4);
				Eigen::MatrixXd p;
				polyfem::autogen::q_nodes_2d(1, p);
				return Eigen::RowVector2d(p(local_index, 0), p(local_index, 1));
			}

			Eigen::RowVector2d tri_local_node_coordinates(int local_index)
			{
				Eigen::MatrixXd p;
				polyfem::autogen::p_nodes_2d(1, p);
				return Eigen::RowVector2d(p(local_index, 0), p(local_index, 1));
			}

			Eigen::RowVector3d linear_tet_local_node_coordinates(int local_index)
			{
				Eigen::MatrixXd p;
				polyfem::autogen::p_nodes_3d(1, p);
				return Eigen::RowVector3d(p(local_index, 0), p(local_index, 1), p(local_index, 2));
			}

			Eigen::RowVector3d linear_hex_local_node_coordinates(int local_index)
			{
				Eigen::MatrixXd p;
				polyfem::autogen::q_nodes_3d(1, p);
				return Eigen::RowVector3d(p(local_index, 0), p(local_index, 1), p(local_index, 2));
			}

			Eigen::Matrix2d quad_local_node_coordinates_from_edge(int le)
			{
				Eigen::Matrix2d res(2, 2);
				res.row(0) = quad_local_node_coordinates(le);
				res.row(1) = quad_local_node_coordinates((le + 1) % 4);

				return res;
			}

			Eigen::Matrix2d tri_local_node_coordinates_from_edge(int le)
			{
				Eigen::Matrix2d res(2, 2);
				res.row(0) = tri_local_node_coordinates(le);
				res.row(1) = tri_local_node_coordinates((le + 1) % 3);

				return res;
			}
		} // namespace

		Eigen::MatrixXd utils::BoundarySampler::tet_local_node_coordinates_from_face(int lf)
		{
			Eigen::Matrix<int, 4, 3> fv;
			fv.row(0) << 0, 1, 2;
			fv.row(1) << 0, 1, 3;
			fv.row(2) << 1, 2, 3;
			fv.row(3) << 2, 0, 3;

			Eigen::MatrixXd res(3, 3);
			for (int i = 0; i < 3; ++i)
				res.row(i) = linear_tet_local_node_coordinates(fv(lf, i));

			return res;
		}

		Eigen::MatrixXd utils::BoundarySampler::hex_local_node_coordinates_from_face(int lf)
		{
			Eigen::Matrix<int, 6, 4> fv;
			fv.row(0) << 0, 3, 7, 4;
			fv.row(1) << 1, 2, 6, 5;
			fv.row(2) << 0, 1, 5, 4;
			fv.row(3) << 3, 2, 6, 7;
			fv.row(4) << 0, 1, 2, 3;
			fv.row(5) << 4, 5, 6, 7;

			Eigen::MatrixXd res(4, 3);
			for (int i = 0; i < 4; ++i)
				res.row(i) = linear_hex_local_node_coordinates(fv(lf, i));

			return res;
		}

		void utils::BoundarySampler::quadrature_for_quad_edge(int index, int order, int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
		{
			auto endpoints = quad_local_node_coordinates_from_edge(index);

			Quadrature quad;
			LineQuadrature quad_rule;
			quad_rule.get_quadrature(order, quad);

			points.resize(quad.points.rows(), endpoints.cols());
			uv.resize(quad.points.rows(), 2);

			uv.col(0) = (1.0 - quad.points.array());
			uv.col(1) = quad.points.array();

			for (int c = 0; c < 2; ++c)
			{
				points.col(c) = (1.0 - quad.points.array()) * endpoints(0, c) + quad.points.array() * endpoints(1, c);
			}

			weights = quad.weights;
			weights *= mesh.edge_length(gid);
		}

		void utils::BoundarySampler::quadrature_for_tri_edge(int index, int order, int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
		{
			auto endpoints = tri_local_node_coordinates_from_edge(index);

			Quadrature quad;
			LineQuadrature quad_rule;
			quad_rule.get_quadrature(order, quad);

			points.resize(quad.points.rows(), endpoints.cols());
			uv.resize(quad.points.rows(), 2);

			uv.col(0) = (1.0 - quad.points.array());
			uv.col(1) = quad.points.array();

			for (int c = 0; c < 2; ++c)
			{
				points.col(c) = (1.0 - quad.points.array()) * endpoints(0, c) + quad.points.array() * endpoints(1, c);
			}

			weights = quad.weights;
			weights *= mesh.edge_length(gid);
		}

		void utils::BoundarySampler::quadrature_for_quad_face(int index, int order, int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
		{
			auto endpoints = hex_local_node_coordinates_from_face(index);

			Quadrature quad;
			QuadQuadrature quad_rule;
			quad_rule.get_quadrature(order, quad);

			const int n_pts = quad.points.rows();
			points.resize(n_pts, endpoints.cols());

			uv.resize(quad.points.rows(), 4);
			uv.col(0) = quad.points.col(0);
			uv.col(1) = 1 - uv.col(0).array();
			uv.col(2) = quad.points.col(1);
			uv.col(3) = 1 - uv.col(2).array();

			for (int i = 0; i < n_pts; ++i)
			{
				const double b1 = quad.points(i, 0);
				const double b2 = 1 - b1;

				const double b3 = quad.points(i, 1);
				const double b4 = 1 - b3;

				for (int c = 0; c < 3; ++c)
				{
					points(i, c) = b3 * (b1 * endpoints(0, c) + b2 * endpoints(1, c)) + b4 * (b1 * endpoints(3, c) + b2 * endpoints(2, c));
				}
			}

			weights = quad.weights;
			weights *= mesh.quad_area(gid);
		}

		void utils::BoundarySampler::quadrature_for_tri_face(int index, int order, int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
		{
			auto endpoints = tet_local_node_coordinates_from_face(index);
			Quadrature quad;
			TriQuadrature quad_rule;
			quad_rule.get_quadrature(order, quad);

			const int n_pts = quad.points.rows();
			points.resize(n_pts, endpoints.cols());

			uv.resize(quad.points.rows(), 3);
			uv.col(0) = quad.points.col(0);
			uv.col(1) = quad.points.col(1);
			uv.col(2) = 1 - uv.col(0).array() - uv.col(1).array();

			for (int i = 0; i < n_pts; ++i)
			{
				const double b1 = quad.points(i, 0);
				const double b3 = quad.points(i, 1);
				const double b2 = 1 - b1 - b3;

				for (int c = 0; c < 3; ++c)
				{
					points(i, c) = b1 * endpoints(0, c) + b2 * endpoints(1, c) + b3 * endpoints(2, c);
				}
			}

			weights = quad.weights;
			// 2 * because weights sum to 1/2 already
			weights *= 2 * mesh.tri_area(gid);
		}

		void utils::BoundarySampler::sample_parametric_quad_edge(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
		{
			auto endpoints = quad_local_node_coordinates_from_edge(index);
			const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
			samples.resize(n_samples, endpoints.cols());

			uv.resize(n_samples, 2);

			uv.col(0) = (1.0 - t.array());
			uv.col(1) = t.array();

			for (int c = 0; c < 2; ++c)
			{
				samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
			}
		}

		void utils::BoundarySampler::sample_parametric_tri_edge(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
		{
			auto endpoints = tri_local_node_coordinates_from_edge(index);
			const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
			samples.resize(n_samples, endpoints.cols());

			uv.resize(n_samples, 2);

			uv.col(0) = (1.0 - t.array());
			uv.col(1) = t.array();

			for (int c = 0; c < 2; ++c)
			{
				samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
			}
		}

		void utils::BoundarySampler::sample_parametric_quad_face(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
		{
			auto endpoints = hex_local_node_coordinates_from_face(index);
			const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
			samples.resize(n_samples * n_samples, endpoints.cols());
			Eigen::MatrixXd left(n_samples, endpoints.cols());
			Eigen::MatrixXd right(n_samples, endpoints.cols());

			uv.resize(n_samples * n_samples, endpoints.cols());
			uv.setZero();

			for (int c = 0; c < 3; ++c)
			{
				left.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
				right.col(c) = (1.0 - t.array()).matrix() * endpoints(3, c) + t * endpoints(2, c);
			}
			for (int c = 0; c < 3; ++c)
			{
				Eigen::MatrixXd x = (1.0 - t.array()).matrix() * left.col(c).transpose() + t * right.col(c).transpose();
				assert(x.size() == n_samples * n_samples);

				samples.col(c) = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
			}
		}

		void utils::BoundarySampler::sample_parametric_tri_face(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
		{
			auto endpoints = tet_local_node_coordinates_from_face(index);
			const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
			samples.resize(n_samples * n_samples, endpoints.cols());

			uv.resize(n_samples * n_samples, 3);

			int counter = 0;
			for (int u = 0; u < n_samples; ++u)
			{
				for (int v = 0; v < n_samples; ++v)
				{
					if (t(u) + t(v) > 1)
						continue;

					uv(counter, 0) = t(u);
					uv(counter, 1) = t(v);
					uv(counter, 2) = 1 - t(u) - t(v);

					for (int c = 0; c < 3; ++c)
					{
						samples(counter, c) = t(u) * endpoints(0, c) + t(v) * endpoints(1, c) + (1 - t(u) - t(v)) * endpoints(2, c);
					}
					++counter;
				}
			}
			samples.conservativeResize(counter, 3);
			uv.conservativeResize(counter, 3);
		}

		void utils::BoundarySampler::sample_polygon_edge(int face_id, int edge_id, int n_samples, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
		{
			assert(!mesh.is_volume());

			const CMesh2D &mesh2d = dynamic_cast<const CMesh2D &>(mesh);

			auto index = mesh2d.get_index_from_face(face_id);

			bool found = false;
			for (int i = 0; i < mesh2d.n_face_vertices(face_id); ++i)
			{
				if (index.edge == edge_id)
				{
					found = true;
					break;
				}

				index = mesh2d.next_around_face(index);
			}

			assert(found);

			auto p0 = mesh2d.point(index.vertex);
			auto p1 = mesh2d.point(mesh2d.switch_edge(index).vertex);
			const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
			samples.resize(n_samples, p0.cols());

			uv.resize(n_samples, 2);

			uv.col(0) = (1.0 - t.array());
			uv.col(1) = t.array();

			for (int c = 0; c < 2; ++c)
			{
				samples.col(c) = (1.0 - t.array()) * p0(c) + t.array() * p1(c);
			}
		}

		void utils::BoundarySampler::quadrature_for_polygon_edge(int face_id, int edge_id, int order, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
		{
			assert(!mesh.is_volume());

			const CMesh2D &mesh2d = dynamic_cast<const CMesh2D &>(mesh);

			auto index = mesh2d.get_index_from_face(face_id);

			bool found = false;
			for (int i = 0; i < mesh2d.n_face_vertices(face_id); ++i)
			{
				if (index.edge == edge_id)
				{
					found = true;
					break;
				}

				index = mesh2d.next_around_face(index);
			}

			assert(found);

			auto p0 = mesh2d.point(index.vertex);
			auto p1 = mesh2d.point(mesh2d.switch_edge(index).vertex);
			Quadrature quad;
			LineQuadrature quad_rule;
			quad_rule.get_quadrature(order, quad);

			points.resize(quad.points.rows(), p0.cols());
			uv.resize(quad.points.rows(), 2);

			uv.col(0) = (1.0 - quad.points.array());
			uv.col(1) = quad.points.array();

			for (int c = 0; c < 2; ++c)
			{
				points.col(c) = (1.0 - quad.points.array()) * p0(c) + quad.points.array() * p1(c);
			}

			weights = quad.weights;
			weights *= mesh.edge_length(edge_id);
		}

		void utils::BoundarySampler::normal_for_quad_edge(int index, Eigen::MatrixXd &normal)
		{
			auto endpoints = quad_local_node_coordinates_from_edge(index);
			const Eigen::MatrixXd e = endpoints.row(0) - endpoints.row(1);
			normal.resize(1, 2);
			normal(0) = -e(1);
			normal(1) = e(0);
			normal.normalize();
		}

		void utils::BoundarySampler::normal_for_tri_edge(int index, Eigen::MatrixXd &normal)
		{
			auto endpoints = tri_local_node_coordinates_from_edge(index);
			const Eigen::MatrixXd e = endpoints.row(0) - endpoints.row(1);
			normal.resize(1, 2);
			normal(0) = -e(1);
			normal(1) = e(0);
			normal.normalize();
		}

		void utils::BoundarySampler::normal_for_quad_face(int index, Eigen::MatrixXd &normal)
		{
			auto endpoints = hex_local_node_coordinates_from_face(index);
			const Eigen::RowVector3d e1 = endpoints.row(0) - endpoints.row(1);
			const Eigen::RowVector3d e2 = endpoints.row(0) - endpoints.row(2);
			normal = e1.cross(e2);
			normal.normalize();

			if (index == 0 || index == 3 || index == 4)
				normal *= -1;
		}

		void utils::BoundarySampler::normal_for_tri_face(int index, Eigen::MatrixXd &normal)
		{
			auto endpoints = tet_local_node_coordinates_from_face(index);
			const Eigen::RowVector3d e1 = endpoints.row(0) - endpoints.row(1);
			const Eigen::RowVector3d e2 = endpoints.row(0) - endpoints.row(2);
			normal = e1.cross(e2);
			normal.normalize();

			if (index == 0)
				normal *= -1;
		}

		void utils::BoundarySampler::normal_for_polygon_edge(int face_id, int edge_id, const Mesh &mesh, Eigen::MatrixXd &normal)
		{
			assert(!mesh.is_volume());

			const CMesh2D &mesh2d = dynamic_cast<const CMesh2D &>(mesh);

			auto index = mesh2d.get_index_from_face(face_id);

			bool found = false;
			for (int i = 0; i < mesh2d.n_face_vertices(face_id); ++i)
			{
				if (index.edge == edge_id)
				{
					found = true;
					break;
				}

				index = mesh2d.next_around_face(index);
			}

			assert(found);

			auto p0 = mesh2d.point(index.vertex);
			auto p1 = mesh2d.point(mesh2d.switch_edge(index).vertex);
			const Eigen::MatrixXd e = p0 - p1;
			normal.resize(1, 2);
			normal(0) = -e(1);
			normal(1) = e(0);
			normal.normalize();
		}

		bool utils::BoundarySampler::boundary_quadrature(const LocalBoundary &local_boundary, const int order, const Mesh &mesh, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::MatrixXd &normals, Eigen::VectorXd &weights, Eigen::VectorXi &global_primitive_ids)
		{
			uv.resize(0, 0);
			points.resize(0, 0);
			normals.resize(0, 0);
			weights.resize(0);
			global_primitive_ids.resize(0);

			for (int i = 0; i < local_boundary.size(); ++i)
			{
				const int gid = local_boundary.global_primitive_id(i);
				Eigen::MatrixXd tmp_p, tmp_uv, tmp_n;
				Eigen::VectorXd tmp_w;
				switch (local_boundary.type())
				{
				case BoundaryType::TRI_LINE:
					quadrature_for_tri_edge(local_boundary[i], order, gid, mesh, tmp_uv, tmp_p, tmp_w);
					normal_for_tri_edge(local_boundary[i], tmp_n);
					break;
				case BoundaryType::QUAD_LINE:
					quadrature_for_quad_edge(local_boundary[i], order, gid, mesh, tmp_uv, tmp_p, tmp_w);
					normal_for_quad_edge(local_boundary[i], tmp_n);
					break;
				case BoundaryType::QUAD:
					quadrature_for_quad_face(local_boundary[i], order, gid, mesh, tmp_uv, tmp_p, tmp_w);
					normal_for_quad_face(local_boundary[i], tmp_n);
					break;
				case BoundaryType::TRI:
					quadrature_for_tri_face(local_boundary[i], order, gid, mesh, tmp_uv, tmp_p, tmp_w);
					normal_for_tri_face(local_boundary[i], tmp_n);
					break;
				case BoundaryType::POLYGON:
					quadrature_for_polygon_edge(local_boundary.element_id(), local_boundary.global_primitive_id(i), order, mesh, tmp_uv, tmp_p, tmp_w);
					normal_for_polygon_edge(local_boundary.element_id(), local_boundary.global_primitive_id(i), mesh, tmp_n);
					break;
				case BoundaryType::INVALID:
					assert(false);
					break;
				default:
					assert(false);
				}

				uv.conservativeResize(uv.rows() + tmp_uv.rows(), tmp_uv.cols());
				uv.bottomRows(tmp_uv.rows()) = tmp_uv;

				points.conservativeResize(points.rows() + tmp_p.rows(), tmp_p.cols());
				points.bottomRows(tmp_p.rows()) = tmp_p;

				normals.conservativeResize(normals.rows() + tmp_p.rows(), tmp_p.cols());
				for (int k = normals.rows() - tmp_p.rows(); k < normals.rows(); ++k)
					normals.row(k) = tmp_n;

				weights.conservativeResize(weights.rows() + tmp_w.rows(), tmp_w.cols());
				weights.bottomRows(tmp_w.rows()) = tmp_w;

				global_primitive_ids.conservativeResize(global_primitive_ids.rows() + tmp_p.rows());
				global_primitive_ids.bottomRows(tmp_p.rows()).setConstant(gid);
			}

			assert(uv.rows() == global_primitive_ids.size());
			assert(points.rows() == global_primitive_ids.size());
			assert(normals.rows() == global_primitive_ids.size());
			assert(weights.size() == global_primitive_ids.size());

			return true;
		}

		bool utils::BoundarySampler::sample_boundary(const LocalBoundary &local_boundary, const int n_samples, const Mesh &mesh, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples, Eigen::VectorXi &global_primitive_ids)
		{
			uv.resize(0, 0);
			samples.resize(0, 0);
			global_primitive_ids.resize(0);

			for (int i = 0; i < local_boundary.size(); ++i)
			{
				Eigen::MatrixXd tmp, tmp_uv;
				switch (local_boundary.type())
				{
				case BoundaryType::TRI_LINE:
					sample_parametric_tri_edge(local_boundary[i], n_samples, tmp_uv, tmp);
					break;
				case BoundaryType::QUAD_LINE:
					sample_parametric_quad_edge(local_boundary[i], n_samples, tmp_uv, tmp);
					break;
				case BoundaryType::QUAD:
					sample_parametric_quad_face(local_boundary[i], n_samples, tmp_uv, tmp);
					break;
				case BoundaryType::TRI:
					sample_parametric_tri_face(local_boundary[i], n_samples, tmp_uv, tmp);
					break;
				case BoundaryType::POLYGON:
					sample_polygon_edge(local_boundary.element_id(), local_boundary.global_primitive_id(i), n_samples, mesh, tmp_uv, tmp);
					break;
				case BoundaryType::INVALID:
					assert(false);
					break;
				default:
					assert(false);
				}

				uv.conservativeResize(uv.rows() + tmp_uv.rows(), tmp_uv.cols());
				uv.bottomRows(tmp_uv.rows()) = tmp_uv;

				samples.conservativeResize(samples.rows() + tmp.rows(), tmp.cols());
				samples.bottomRows(tmp.rows()) = tmp;

				global_primitive_ids.conservativeResize(global_primitive_ids.rows() + tmp.rows());
				global_primitive_ids.bottomRows(tmp.rows()).setConstant(local_boundary.global_primitive_id(i));
			}

			assert(uv.rows() == global_primitive_ids.size());
			assert(samples.rows() == global_primitive_ids.size());

			return true;
		}

		bool utils::BoundarySampler::boundary_quadrature(const mesh::LocalBoundary &local_boundary, const int order, const mesh::Mesh &mesh, const int i, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::MatrixXd &normals, Eigen::VectorXd &weights)
		{
			assert (local_boundary.size() > i);
				
			uv.resize(0, 0);
			points.resize(0, 0);
			weights.resize(0);
			const int gid = local_boundary.global_primitive_id(i);

			Eigen::MatrixXd normal;
			switch (local_boundary.type())
			{
			case BoundaryType::TRI_LINE:
				quadrature_for_tri_edge(local_boundary[i], order, gid, mesh, uv, points, weights);
				normal_for_tri_edge(local_boundary[i], normal);
				break;
			case BoundaryType::QUAD_LINE:
				quadrature_for_quad_edge(local_boundary[i], order, gid, mesh, uv, points, weights);
				normal_for_quad_edge(local_boundary[i], normal);
				break;
			case BoundaryType::QUAD:
				quadrature_for_quad_face(local_boundary[i], order, gid, mesh, uv, points, weights);
				normal_for_quad_face(local_boundary[i], normal);
				break;
			case BoundaryType::TRI:
				quadrature_for_tri_face(local_boundary[i], order, gid, mesh, uv, points, weights);
				normal_for_tri_face(local_boundary[i], normal);
				break;
			case BoundaryType::POLYGON:
				quadrature_for_polygon_edge(local_boundary.element_id(), gid, order, mesh, uv, points, weights);
				normal_for_polygon_edge(local_boundary.element_id(), gid, mesh, normal);
				break;
			case BoundaryType::INVALID:
				assert(false);
				break;
			default:
				assert(false);
			}

			normals.resize(points.rows(), normal.size());
			for (int k = 0; k < normals.rows(); ++k)
				normals.row(k) = normal;

			return true;
		}
	} // namespace utils
} // namespace polyfem
