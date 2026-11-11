#include "PolygonMesh.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stack>

namespace polyfem::mesh
{
	void PolygonMesh::clear()
	{
		points_.clear();
		facet_ptr_.assign(1, 0);
		corner_vertex_.clear();
		corner_adjacent_facet_.clear();
		edges_.clear();
		corner_edge.clear();
		boundary_edge.clear();
		boundary_vertex.clear();
	}

	int PolygonMesh::create_vertices(const int n)
	{
		const int first = n_vertices();
		points_.resize(first + n);
		boundary_vertex.resize(points_.size(), false);
		return first;
	}

	int PolygonMesh::create_vertex(const Point &p)
	{
		const int v = create_vertices(1);
		points_[v] = p;
		return v;
	}

	void PolygonMesh::bounding_box(Point &min, Point &max) const
	{
		for (int c = 0; c < 3; ++c)
		{
			min[c] = std::numeric_limits<double>::max();
			max[c] = std::numeric_limits<double>::lowest();
		}
		for (const Point &p : points_)
		{
			for (int c = 0; c < 3; ++c)
			{
				min[c] = std::min(min[c], p[c]);
				max[c] = std::max(max[c], p[c]);
			}
		}
	}

	int PolygonMesh::create_polygon(const std::vector<int> &vertices)
	{
		const int f = n_facets();
		for (const int v : vertices)
		{
			assert(v >= 0 && v < n_vertices());
			corner_vertex_.push_back(v);
			corner_adjacent_facet_.push_back(-1);
		}
		corner_edge.resize(corner_vertex_.size(), 0);
		facet_ptr_.push_back(corner_vertex_.size());
		return f;
	}

	void PolygonMesh::flip_facet(const int f)
	{
		const int d = n_facet_vertices(f);
		const int c0 = corners_begin(f);
		const std::vector<int> vertices(corner_vertex_.begin() + c0, corner_vertex_.begin() + c0 + d);
		const std::vector<int> adjacent(corner_adjacent_facet_.begin() + c0, corner_adjacent_facet_.begin() + c0 + d);
		for (int i = 0; i < d; ++i)
		{
			const int i_v = d - 1 - i;
			// The edge starting at the new corner i is the reversed edge that
			// started at the old corner before i_v.
			const int i_f = (i_v == 0) ? d - 1 : i_v - 1;
			corner_vertex_[c0 + i] = vertices[i_v];
			corner_adjacent_facet_[c0 + i] = adjacent[i_f];
		}
	}

	void PolygonMesh::clear_facets()
	{
		facet_ptr_.assign(1, 0);
		corner_vertex_.clear();
		corner_adjacent_facet_.clear();
		corner_edge.clear();
	}

	void PolygonMesh::delete_facets(const std::vector<bool> &to_delete)
	{
		assert(to_delete.size() == n_facets());

		std::vector<int> facets_old2new(n_facets(), -1);
		std::vector<int> new_facet_ptr = {0};
		int new_nb_corners = 0;
		for (int f = 0; f < n_facets(); ++f)
		{
			if (to_delete[f])
				continue;
			facets_old2new[f] = new_facet_ptr.size() - 1;
			for (int c = corners_begin(f); c < corners_end(f); ++c)
			{
				corner_vertex_[new_nb_corners] = corner_vertex_[c];
				corner_adjacent_facet_[new_nb_corners] = corner_adjacent_facet_[c];
				corner_edge[new_nb_corners] = corner_edge[c];
				++new_nb_corners;
			}
			new_facet_ptr.push_back(new_nb_corners);
		}
		corner_vertex_.resize(new_nb_corners);
		corner_adjacent_facet_.resize(new_nb_corners);
		corner_edge.resize(new_nb_corners);
		facet_ptr_ = std::move(new_facet_ptr);

		for (int &f : corner_adjacent_facet_)
		{
			if (f != -1)
				f = facets_old2new[f];
		}
	}

	void PolygonMesh::connect_facets()
	{
		// Chain the corners around each vertex.
		std::vector<int> next_corner_around_vertex(n_corners(), -1);
		std::vector<int> v2c(n_vertices(), -1);
		std::vector<int> c2f(n_corners(), -1);
		for (int f = 0; f < n_facets(); ++f)
		{
			for (int c = corners_begin(f); c < corners_end(f); ++c)
			{
				const int v = corner_vertex_[c];
				next_corner_around_vertex[c] = v2c[v];
				v2c[v] = c;
				c2f[c] = f;
			}
		}

		// Glue each free corner c1 (edge v1 -> v2) to an earlier corner c2 at v1
		// whose facet has the opposite edge v2 -> v1.
		for (int f1 = 0; f1 < n_facets(); ++f1)
		{
			for (int c1 = corners_begin(f1); c1 < corners_end(f1); ++c1)
			{
				if (corner_adjacent_facet_[c1] != -1)
					continue;
				const int v2 = corner_vertex_[next_corner_around_facet(f1, c1)];
				// The chain only links each corner to earlier ones, so c2 != c1.
				for (int c2 = next_corner_around_vertex[c1]; c2 != -1; c2 = next_corner_around_vertex[c2])
				{
					const int f2 = c2f[c2];
					const int c3 = prev_corner_around_facet(f2, c2);
					if (corner_vertex_[c3] == v2)
					{
						corner_adjacent_facet_[c1] = f2;
						corner_adjacent_facet_[c3] = f1;
						break;
					}
				}
			}
		}
	}

	int PolygonMesh::connected_components(std::vector<int> &component) const
	{
		int nb_components = 0;
		component.assign(n_facets(), -1);
		for (int f = 0; f < n_facets(); ++f)
		{
			if (component[f] != -1)
				continue;
			std::stack<int> S;
			S.push(f);
			component[f] = nb_components;
			do
			{
				const int cur_f = S.top();
				S.pop();
				for (int c = corners_begin(cur_f); c < corners_end(cur_f); ++c)
				{
					const int adj_f = corner_adjacent_facet_[c];
					if (adj_f != -1 && component[adj_f] == -1)
					{
						S.push(adj_f);
						component[adj_f] = nb_components;
					}
				}
			} while (!S.empty());
			nb_components++;
		}
		return nb_components;
	}

	int PolygonMesh::create_edge(const int v0, const int v1)
	{
		edges_.push_back({{v0, v1}});
		boundary_edge.push_back(false);
		return edges_.size() - 1;
	}

	void PolygonMesh::clear_edges()
	{
		edges_.clear();
		boundary_edge.clear();
	}
} // namespace polyfem::mesh
