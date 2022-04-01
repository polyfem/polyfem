#include <polyfem/Mesh2D.hpp>

#include <polyfem/MeshUtils.hpp>

#include <polyfem/Logger.hpp>

#include <igl/barycentric_coordinates.h>

#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_repair.h>

#include <cassert>
#include <array>

namespace polyfem {

	void Mesh2D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const
	{
		p0.resize(n_edges(), 2);
		p1.resize(p0.rows(), p0.cols());

		for (GEO::index_t e = 0; e < n_edges(); ++e)
		{
			const int v0 = edge_vertex(e, 0);
			const int v1 = edge_vertex(e, 1);

			p0.row(e) = point(v0);
			p1.row(e) = point(v1);
		}
	}

	void Mesh2D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const
	{
		int count = 0;
		for (size_t i = 0; i < valid_elements.size(); ++i)
		{
			if (valid_elements[i])
			{
				count += n_face_vertices(i);
			}
		}

		p0.resize(count, 2);
		p1.resize(count, 2);

		count = 0;

		for (size_t i = 0; i < valid_elements.size(); ++i)
		{
			if (!valid_elements[i])
				continue;

			auto index = get_index_from_face(i);
			for (int j = 0; j < n_face_vertices(i); ++j)
			{
				p0.row(count) = point(index.vertex);
				p1.row(count) = point(switch_vertex(index).vertex);

				index = next_around_face(index);
				++count;
			}
		}
	}

	RowVectorNd Mesh2D::face_barycenter(const int face_index) const
	{
		RowVectorNd bary(2);
		bary.setZero();

		const int n_vertices = n_face_vertices(face_index);
		Navigation::Index index = get_index_from_face(face_index);

		for (int lv = 0; lv < n_vertices; ++lv)
		{
			bary += point(index.vertex);
			index = next_around_face(index);
		}
		return bary / n_vertices;
	}

	void Mesh2D::barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coords) const
	{
		assert(is_simplex(el_id));

		auto index = get_index_from_face(el_id);
		const auto A = point(index.vertex);

		index = next_around_face(index);
		const auto B = point(index.vertex);

		index = next_around_face(index);
		const auto C = point(index.vertex);

		igl::barycentric_coordinates(p, A, B, C, coords);
	}

	void Mesh2D::compute_body_ids(const std::function<int(const RowVectorNd &)> &marker)
	{
		body_ids_.resize(n_elements());
		std::fill(body_ids_.begin(), body_ids_.end(), -1);

		for (int e = 0; e < n_elements(); ++e)
		{
			const auto bary = face_barycenter(e);
			body_ids_[e] = marker(bary);
		}
	}

	void Mesh2D::compute_boundary_ids(const double eps)
	{
		boundary_ids_.resize(n_edges());
		std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

		RowVectorNd min_corner, max_corner;
		bounding_box(min_corner, max_corner);

		//implement me properly
		for (int e = 0; e < n_edges(); ++e)
		{
			if (!is_boundary_edge(e))
				continue;

			const auto p = edge_barycenter(e);

			if (fabs(p(0) - min_corner[0]) < eps)
				boundary_ids_[e] = 1;
			else if (fabs(p(1) - min_corner[1]) < eps)
				boundary_ids_[e] = 2;
			else if (fabs(p(0) - max_corner[0]) < eps)
				boundary_ids_[e] = 3;
			else if (fabs(p(1) - max_corner[1]) < eps)
				boundary_ids_[e] = 4;

			else
				boundary_ids_[e] = 7;
		}
	}

	void Mesh2D::compute_boundary_ids(const std::function<int(const RowVectorNd &)> &marker)
	{
		boundary_ids_.resize(n_edges());
		std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

		//implement me properly
		for (int e = 0; e < n_edges(); ++e)
		{
			if (!is_boundary_edge(e))
				continue;

			const auto p = edge_barycenter(e);

			boundary_ids_[e] = marker(p);
		}
	}

	void Mesh2D::compute_boundary_ids(const std::function<int(const RowVectorNd &, bool)> &marker)
	{
		boundary_ids_.resize(n_edges());

		for (int e = 0; e < n_edges(); ++e)
		{
			const bool is_boundary = is_boundary_edge(e);
			const auto p = edge_barycenter(e);
			boundary_ids_[e] = marker(p, is_boundary);
		}
	}

	void Mesh2D::compute_boundary_ids(const std::function<int(const std::vector<int> &, bool)> &marker)
	{
		boundary_ids_.resize(n_edges());

		for (int e = 0; e < n_edges(); ++e)
		{
			bool is_boundary = is_boundary_edge(e);
			std::vector<int> vs = {edge_vertex(e, 0), edge_vertex(e, 1)};
			std::sort(vs.begin(), vs.end());
			boundary_ids_[e] = marker(vs, is_boundary);
		}
	}

	void Mesh2D::elements_boxes(std::vector<std::array<Eigen::Vector3d, 2>> &boxes) const
	{
		boxes.resize(n_elements());

		for (int i = 0; i < n_elements(); ++i)
		{
			auto &box = boxes[i];
			box[0] << std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0;
			box[1] << std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), 0;

			auto index = get_index_from_face(i);

			for (int j = 0; j < n_face_vertices(i); ++j)
			{
				for (int d = 0; d < 2; ++d)
				{
					box[0][d] = std::min(box[0][d], point(index.vertex)[d]);
					box[1][d] = std::max(box[1][d], point(index.vertex)[d]);
				}
				index = next_around_face(index);
			}
		}
	}
}