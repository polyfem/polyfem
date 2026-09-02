////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <Eigen/Geometry>

#include <igl/boundary_facets.h>
#include <igl/oriented_facets.h>
#include <igl/edges.h>

#include <unordered_set>
#include <set>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
namespace polyfem::mesh
{
	using namespace polyfem::utils;

	std::vector<MeshWithID> Mesh::split() const
	{
		std::set<int> ids;
		for (int e = 0; e < n_elements(); ++e)
			ids.insert(get_geometry_id(e));

		std::vector<MeshWithID> result;
		result.reserve(ids.size());
		for (const int id : ids)
		{
			auto child = copy();
			std::vector<bool> keep(n_elements(), false);
			for (int e = 0; e < n_elements(); ++e)
				keep[e] = get_geometry_id(e) == id;
			child->remove_elements(keep);
			result.push_back({id, std::move(child)});
		}
		return result;
	}

	void Mesh::filter_element_data(const std::vector<bool> &keep)
	{
		const int kept = std::count(keep.begin(), keep.end(), true);

		auto filter_vector = [&keep, kept](auto &values) {
			if (values.empty())
				return;
			assert(values.size() == keep.size());
			using Value = typename std::decay_t<decltype(values)>::value_type;
			std::vector<Value> filtered;
			filtered.reserve(kept);
			for (int i = 0; i < keep.size(); ++i)
				if (keep[i])
					filtered.push_back(values[i]);
			values = std::move(filtered);
		};

		filter_vector(elements_tag_);
		filter_vector(body_ids_);
		filter_vector(geometry_ids_);
		filter_vector(cell_weights_);

		if (orders_.size() > 0)
		{
			assert(orders_.rows() == keep.size());
			Eigen::MatrixXi filtered(kept, orders_.cols());
			for (int i = 0, j = 0; i < keep.size(); ++i)
				if (keep[i])
					filtered.row(j++) = orders_.row(i);
			orders_ = std::move(filtered);
		}
	}

	std::unique_ptr<Mesh> Mesh::create(MeshData data, const bool non_conforming)
	{
		data.validate();
		const int dim = data.dimension();
		assert(dim == 2 || dim == 3);

		std::unique_ptr<Mesh> mesh;
		if (dim == 2 && non_conforming)
			mesh = std::make_unique<NCMesh2D>();
		else if (dim == 2 && !non_conforming)
			mesh = std::make_unique<CMesh2D>();
		else if (dim == 3 && non_conforming)
			mesh = std::make_unique<NCMesh3D>();
		else if (dim == 3 && !non_conforming)
			mesh = std::make_unique<CMesh3D>();
		if (!mesh || !mesh->build_from_data(data))
			log_and_throw_error("Unable to construct runtime mesh from MeshData.");
		return mesh;
	}

	bool Mesh::build_from_data(const MeshData &data)
	{
		if (!build_topology(data))
			return false;

		std::vector<int> tmp(data.elements.data(), data.elements.data() + data.elements.size());
		std::sort(tmp.begin(), tmp.end());
		tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
		if (!tmp.empty() && tmp.front() == -1)
			tmp.erase(tmp.begin());

		in_ordered_vertices_ = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(tmp.data(), tmp.size());

		in_ordered_edges_.resize(n_edges(), 2);
		for (int e = 0; e < n_edges(); ++e)
			in_ordered_edges_.row(e) << edge_vertex(e, 0), edge_vertex(e, 1);
		if (dimension() == 2)
			in_ordered_faces_.resize(0, 0);
		else
		{
			int width = 0;
			for (int f = 0; f < n_faces(); ++f)
				width = std::max(width, n_face_vertices(f));
			in_ordered_faces_.setConstant(n_faces(), width, -1);
			for (int f = 0; f < n_faces(); ++f)
				for (int lv = 0; lv < n_face_vertices(f); ++lv)
					in_ordered_faces_(f, lv) = face_vertex(f, lv);
		}

		if (!data.higher_order_connectivity.empty())
			attach_higher_order_nodes(data.higher_order_nodes, data.higher_order_connectivity);
		if (!data.higher_order_weights.empty())
		{
			set_cell_weights(data.higher_order_weights);
			set_is_rational(std::any_of(
				data.higher_order_weights.begin(), data.higher_order_weights.end(),
				[](const auto &weights) { return !weights.empty(); }));
		}
		if (!data.body_ids.empty())
			set_body_ids(data.body_ids);
		if (!data.geometry_ids.empty())
			set_geometry_ids(data.geometry_ids);

		if (!data.boundary_ids.empty())
		{
			std::unordered_map<std::vector<int>, int, HashVector> labels;
			for (int i = 0; i < data.boundary_elements.size(); ++i)
			{
				std::vector<int> side = data.boundary_elements[i];
				std::sort(side.begin(), side.end());
				const auto [it, inserted] = labels.emplace(std::move(side), data.boundary_ids[i]);
				if (!inserted && it->second != data.boundary_ids[i])
					logger().warn("Mesh side has multiple labels; using {}.", it->second);
			}
			compute_boundary_ids([&](const size_t primitive, const std::vector<int> &vertices, const RowVectorNd &, const bool) {
				std::vector<int> side = vertices;
				std::sort(side.begin(), side.end());
				const auto it = labels.find(side);
				return it == labels.end() ? get_default_boundary_id(primitive) : it->second;
			});
		}

		return true;
	}

	////////////////////////////////////////////////////////////////////////////////

	void Mesh::edge_barycenters(Eigen::MatrixXd &barycenters) const
	{
		barycenters.resize(n_edges(), dimension());
		for (int e = 0; e < n_edges(); ++e)
		{
			barycenters.row(e) = edge_barycenter(e);
		}
	}

	void Mesh::face_barycenters(Eigen::MatrixXd &barycenters) const
	{
		barycenters.resize(n_faces(), dimension());
		for (int f = 0; f < n_faces(); ++f)
		{
			barycenters.row(f) = face_barycenter(f);
		}
	}

	void Mesh::cell_barycenters(Eigen::MatrixXd &barycenters) const
	{
		barycenters.resize(n_cells(), dimension());
		for (int c = 0; c < n_cells(); ++c)
		{
			barycenters.row(c) = cell_barycenter(c);
		}
	}

	////////////////////////////////////////////////////////////////////////////////

	// Queries on the tags
	bool Mesh::is_spline_compatible(const int el_id) const
	{
		if (is_volume())
		{
			return elements_tag_[el_id] == ElementType::REGULAR_INTERIOR_CUBE
				   || elements_tag_[el_id] == ElementType::REGULAR_BOUNDARY_CUBE;
			// || elements_tag_[el_id] == ElementType::SIMPLE_SINGULAR_INTERIOR_CUBE
			// || elements_tag_[el_id] == ElementType::SIMPLE_SINGULAR_BOUNDARY_CUBE;
		}
		else
		{
			return elements_tag_[el_id] == ElementType::REGULAR_INTERIOR_CUBE
				   || elements_tag_[el_id] == ElementType::REGULAR_BOUNDARY_CUBE;
			// || elements_tag_[el_id] == ElementType::INTERFACE_CUBE
			// || elements_tag_[el_id] == ElementType::SIMPLE_SINGULAR_INTERIOR_CUBE;
		}
	}

	// -----------------------------------------------------------------------------

	bool Mesh::is_cube(const int el_id) const
	{
		return elements_tag_[el_id] == ElementType::INTERFACE_CUBE
			   || elements_tag_[el_id] == ElementType::REGULAR_INTERIOR_CUBE
			   || elements_tag_[el_id] == ElementType::REGULAR_BOUNDARY_CUBE
			   || elements_tag_[el_id] == ElementType::SIMPLE_SINGULAR_INTERIOR_CUBE
			   || elements_tag_[el_id] == ElementType::SIMPLE_SINGULAR_BOUNDARY_CUBE
			   || elements_tag_[el_id] == ElementType::MULTI_SINGULAR_INTERIOR_CUBE
			   || elements_tag_[el_id] == ElementType::MULTI_SINGULAR_BOUNDARY_CUBE;
	}

	// -----------------------------------------------------------------------------

	bool Mesh::is_polytope(const int el_id) const
	{
		return elements_tag_[el_id] == ElementType::INTERIOR_POLYTOPE
			   || elements_tag_[el_id] == ElementType::BOUNDARY_POLYTOPE;
	}

	void Mesh::update_nodes(const Eigen::VectorXi &in_node_to_node)
	{
		if (in_node_to_node.size() <= 0 || node_ids_.empty())
		{
			node_ids_.clear();
			return;
		}

		const auto tmp = node_ids_;

		for (int n = 0; n < n_vertices(); ++n)
		{
			node_ids_[in_node_to_node[n]] = tmp[n];
		}
	}

	void Mesh::compute_node_ids(const std::function<int(const size_t, const RowVectorNd &, bool)> &marker)
	{
		node_ids_.resize(n_vertices());

		for (int n = 0; n < n_vertices(); ++n)
		{
			bool is_boundary = is_boundary_vertex(n);
			const auto p = point(n);
			node_ids_[n] = marker(n, p, is_boundary);
		}
	}

	bool Mesh::is_simplex(const int el_id) const
	{
		return elements_tag_[el_id] == ElementType::SIMPLEX;
	}

	bool Mesh::is_prism(const int el_id) const
	{
		return elements_tag_[el_id] == ElementType::PRISM;
	}

	bool Mesh::is_pyramid(const int el_id) const
	{
		return elements_tag_[el_id] == ElementType::PYRAMID;
	}

	std::vector<std::pair<int, int>> Mesh::edges() const
	{
		std::vector<std::pair<int, int>> res;
		res.reserve(n_edges());

		for (int e_id = 0; e_id < n_edges(); ++e_id)
		{
			const int e0 = edge_vertex(e_id, 0);
			const int e1 = edge_vertex(e_id, 1);

			res.emplace_back(std::min(e0, e1), std::max(e0, e1));
		}

		return res;
	}

	std::vector<std::vector<int>> Mesh::faces() const
	{
		std::vector<std::vector<int>> res(n_faces());

		for (int f_id = 0; f_id < n_faces(); ++f_id)
		{
			auto &tmp = res[f_id];
			for (int lv_id = 0; lv_id < n_face_vertices(f_id); ++lv_id)
				tmp.push_back(face_vertex(f_id, lv_id));

			std::sort(tmp.begin(), tmp.end());
		}

		return res;
	}

	std::unordered_map<std::pair<int, int>, size_t, HashPair> Mesh::edges_to_ids() const
	{
		std::unordered_map<std::pair<int, int>, size_t, HashPair> res;
		res.reserve(n_edges());

		for (int e_id = 0; e_id < n_edges(); ++e_id)
		{
			const int e0 = edge_vertex(e_id, 0);
			const int e1 = edge_vertex(e_id, 1);

			res[std::pair<int, int>(std::min(e0, e1), std::max(e0, e1))] = e_id;
		}

		return res;
	}

	std::unordered_map<std::vector<int>, size_t, HashVector> Mesh::faces_to_ids() const
	{
		std::unordered_map<std::vector<int>, size_t, HashVector> res;
		res.reserve(n_faces());

		for (int f_id = 0; f_id < n_faces(); ++f_id)
		{
			std::vector<int> f;
			f.reserve(n_face_vertices(f_id));
			for (int lv_id = 0; lv_id < n_face_vertices(f_id); ++lv_id)
				f.push_back(face_vertex(f_id, lv_id));
			std::sort(f.begin(), f.end());

			res[f] = f_id;
		}

		return res;
	}

	void Mesh::append(const Mesh &mesh)
	{
		const int n_vertices = this->n_vertices();

		elements_tag_.insert(elements_tag_.end(), mesh.elements_tag_.begin(), mesh.elements_tag_.end());

		// --------------------------------------------------------------------

		// Initialize node_ids_ if it is not initialized yet.
		if (!has_node_ids() && mesh.has_node_ids())
		{
			node_ids_.resize(n_vertices);
			for (int i = 0; i < node_ids_.size(); ++i)
				node_ids_[i] = get_node_id(i); // results in default if node_ids_ is empty
		}

		if (mesh.has_node_ids())
		{
			node_ids_.insert(node_ids_.end(), mesh.node_ids_.begin(), mesh.node_ids_.end());
		}
		else if (has_node_ids()) // && !mesh.has_node_ids()
		{
			node_ids_.resize(n_vertices + mesh.n_vertices());
			for (int i = 0; i < mesh.n_vertices(); ++i)
				node_ids_[n_vertices + i] = mesh.get_node_id(i); // results in default if node_ids_ is empty
		}

		assert(node_ids_.empty() || node_ids_.size() == n_vertices + mesh.n_vertices());

		// --------------------------------------------------------------------

		// Initialize boundary_ids_ if it is not initialized yet.
		if (!has_boundary_ids() && mesh.has_boundary_ids())
		{
			boundary_ids_.resize(n_boundary_elements());
			for (int i = 0; i < boundary_ids_.size(); ++i)
				boundary_ids_[i] = get_default_boundary_id(i);
		}

		if (mesh.has_boundary_ids())
		{
			boundary_ids_.insert(boundary_ids_.end(), mesh.boundary_ids_.begin(), mesh.boundary_ids_.end());
		}
		else if (has_boundary_ids()) // && !mesh.has_boundary_ids()
		{
			boundary_ids_.resize(n_boundary_elements() + mesh.n_boundary_elements());
			for (int i = 0; i < mesh.n_boundary_elements(); ++i)
				boundary_ids_[n_boundary_elements() + i] = mesh.get_boundary_id(i); // results in default if mesh.boundary_ids_ is empty
		}

		// --------------------------------------------------------------------

		// Initialize body_ids_ if it is not initialized yet.
		if (!has_body_ids() && mesh.has_body_ids())
			body_ids_ = std::vector<int>(n_elements(), 0); // 0 is the default body_id

		if (mesh.has_body_ids())
			body_ids_.insert(body_ids_.end(), mesh.body_ids_.begin(), mesh.body_ids_.end());
		else if (has_body_ids())                                   // && !mesh.has_body_ids()
			body_ids_.resize(n_elements() + mesh.n_elements(), 0); // 0 is the default body_id

		// --------------------------------------------------------------------
		// Initialize geometry_ids_ if it is not initialized yet.
		if (!has_geometry_ids() && mesh.has_geometry_ids())
			geometry_ids_ = std::vector<int>(n_elements(), 0);

		if (mesh.has_geometry_ids())
			geometry_ids_.insert(geometry_ids_.end(), mesh.geometry_ids_.begin(), mesh.geometry_ids_.end());
		else if (has_geometry_ids())
			geometry_ids_.resize(n_elements() + mesh.n_elements(), 0);

		// --------------------------------------------------------------------

		if (orders_.size() == 0)
			orders_.setOnes(n_elements(), 1);
		Eigen::MatrixXi mesh_orders = mesh.orders_;
		if (mesh_orders.size() == 0)
			mesh_orders.setOnes(mesh.n_elements(), 1);
		assert(orders_.cols() == mesh_orders.cols());
		orders_.conservativeResize(orders_.rows() + mesh_orders.rows(), orders_.cols());
		orders_.bottomRows(mesh_orders.rows()) = mesh_orders;

		is_rational_ = is_rational_ || mesh.is_rational_;

		// --------------------------------------------------------------------
		for (const auto &n : mesh.edge_nodes_)
		{
			auto tmp = n;
			tmp.v1 += n_vertices;
			tmp.v2 += n_vertices;
			edge_nodes_.push_back(tmp);
		}
		for (const auto &n : mesh.face_nodes_)
		{
			auto tmp = n;
			tmp.v1 += n_vertices;
			tmp.v2 += n_vertices;
			tmp.v3 += n_vertices;
			face_nodes_.push_back(tmp);
		}
		for (const auto &n : mesh.cell_nodes_)
		{
			auto tmp = n;
			tmp.v1 += n_vertices;
			tmp.v2 += n_vertices;
			tmp.v3 += n_vertices;
			tmp.v4 += n_vertices;
			cell_nodes_.push_back(tmp);
		}
		cell_weights_.insert(cell_weights_.end(), mesh.cell_weights_.begin(), mesh.cell_weights_.end());
		// --------------------------------------------------------------------

		assert(in_ordered_vertices_.cols() == mesh.in_ordered_vertices_.cols());
		in_ordered_vertices_.conservativeResize(in_ordered_vertices_.rows() + mesh.in_ordered_vertices_.rows(), in_ordered_vertices_.cols());
		in_ordered_vertices_.bottomRows(mesh.in_ordered_vertices_.rows()) = mesh.in_ordered_vertices_.array() + n_vertices;

		if (in_ordered_edges_.size() == 0 || mesh.in_ordered_edges_.size() == 0)
			in_ordered_edges_.resize(0, 0);
		else
		{
			assert(in_ordered_edges_.cols() == mesh.in_ordered_edges_.cols());
			utils::append_rows(in_ordered_edges_, mesh.in_ordered_edges_.array() + n_vertices);
		}

		if (in_ordered_faces_.size() == 0 || mesh.in_ordered_faces_.size() == 0)
			in_ordered_faces_.resize(0, 0);
		else
		{
			assert(in_ordered_faces_.cols() == mesh.in_ordered_faces_.cols());
			utils::append_rows(in_ordered_faces_, mesh.in_ordered_faces_.array() + n_vertices);
		}
	}

	namespace
	{
		template <typename T>
		void transform_high_order_nodes(std::vector<T> &nodes, const MatrixNd &A, const VectorNd &b)
		{
			for (T &n : nodes)
			{
				if (n.nodes.size())
				{
					n.nodes = (n.nodes * A.transpose()).rowwise() + b.transpose();
				}
			}
		}
	} // namespace

	void Mesh::apply_affine_transformation(const MatrixNd &A, const VectorNd &b)
	{
		for (int i = 0; i < n_vertices(); ++i)
		{
			VectorNd p = point(i).transpose();
			p = A * p + b;
			set_point(i, p.transpose());
		}

		transform_high_order_nodes(edge_nodes_, A, b);
		transform_high_order_nodes(face_nodes_, A, b);
		transform_high_order_nodes(cell_nodes_, A, b);
	}
} // namespace polyfem::mesh
