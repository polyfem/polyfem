////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>

#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/MshReader.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_geometry.h>

#include <Eigen/Geometry>

#include <igl/boundary_facets.h>
#include <igl/oriented_facets.h>
#include <igl/edges.h>

#include <filesystem>
#include <unordered_set>

////////////////////////////////////////////////////////////////////////////////
namespace polyfem::mesh
{
	using namespace polyfem::io;
	using namespace polyfem::utils;

	namespace
	{
		std::vector<int> sort_face(const Eigen::RowVectorXi f)
		{
			std::vector<int> sorted_face(f.data(), f.data() + f.size());
			std::sort(sorted_face.begin(), sorted_face.end());
			return sorted_face;
		}

		// Constructs a list of unique faces represented in a given mesh (V,T)
		//
		// Inputs:
		//   T: #T × 4 matrix of indices of tet corners
		// Outputs:
		//   F: #F × 3 list of faces in no particular order
		template <typename DerivedT, typename DerivedF>
		void get_faces(
			const Eigen::MatrixBase<DerivedT> &T,
			Eigen::PlainObjectBase<DerivedF> &F)
		{
			assert(T.cols() == 4);
			assert(T.rows() >= 1);

			Eigen::MatrixXi BF, OF;
			igl::boundary_facets(T, BF);
			igl::oriented_facets(T, OF); // boundary facets + duplicated interior faces
			assert((OF.rows() + BF.rows()) % 2 == 0);
			const int num_faces = (OF.rows() + BF.rows()) / 2;
			F.resize(num_faces, 3);
			F.topRows(BF.rows()) = BF;
			std::unordered_set<std::vector<int>, HashVector> processed_faces;
			for (int fi = 0; fi < BF.rows(); fi++)
			{
				processed_faces.insert(sort_face(BF.row(fi)));
			}

			for (int fi = 0; fi < OF.rows(); fi++)
			{
				std::vector<int> sorted_face = sort_face(OF.row(fi));
				const auto iter = processed_faces.find(sorted_face);
				if (iter == processed_faces.end())
				{
					F.row(processed_faces.size()) = OF.row(fi);
					processed_faces.insert(sorted_face);
				}
			}

			assert(F.rows() == processed_faces.size());
		}
	} // namespace

	std::unique_ptr<Mesh> Mesh::create(const int dim, const bool non_conforming)
	{
		assert(dim == 2 || dim == 3);
		if (dim == 2 && non_conforming)
			return std::make_unique<NCMesh2D>();
		else if (dim == 2 && !non_conforming)
			return std::make_unique<CMesh2D>();
		else if (dim == 3 && non_conforming)
			return std::make_unique<NCMesh3D>();
		else if (dim == 3 && !non_conforming)
			return std::make_unique<CMesh3D>();
		throw std::runtime_error("Invalid dimension");
	}

	std::unique_ptr<Mesh> Mesh::create(GEO::Mesh &meshin, const bool non_conforming)
	{
		if (is_planar(meshin))
		{
			generate_edges(meshin);
			std::unique_ptr<Mesh> mesh = create(2, non_conforming);
			if (mesh->load(meshin))
			{
				mesh->in_ordered_vertices_ = Eigen::VectorXi::LinSpaced(meshin.vertices.nb(), 0, meshin.vertices.nb() - 1);
				assert(mesh->in_ordered_vertices_[0] == 0);
				assert(mesh->in_ordered_vertices_[1] == 1);
				assert(mesh->in_ordered_vertices_[2] == 2);
				assert(mesh->in_ordered_vertices_[mesh->in_ordered_vertices_.size() - 1] == meshin.vertices.nb() - 1);

				mesh->in_ordered_edges_.resize(meshin.edges.nb(), 2);

				for (int e = 0; e < (int)meshin.edges.nb(); ++e)
				{
					for (int lv = 0; lv < 2; ++lv)
					{
						mesh->in_ordered_edges_(e, lv) = meshin.edges.vertex(e, lv);
					}
					assert(mesh->in_ordered_edges_(e, 0) != mesh->in_ordered_edges_(e, 1));
				}
				assert(mesh->in_ordered_edges_.size() > 0);

				mesh->in_ordered_faces_.resize(0, 0);

				return mesh;
			}
		}
		else
		{
			std::unique_ptr<Mesh> mesh = create(3, non_conforming);
			meshin.cells.connect();
			if (mesh->load(meshin))
			{
				mesh->in_ordered_vertices_ = Eigen::VectorXi::LinSpaced(meshin.vertices.nb(), 0, meshin.vertices.nb() - 1);
				assert(mesh->in_ordered_vertices_[0] == 0);
				assert(mesh->in_ordered_vertices_[1] == 1);
				assert(mesh->in_ordered_vertices_[2] == 2);
				assert(mesh->in_ordered_vertices_[mesh->in_ordered_vertices_.size() - 1] == meshin.vertices.nb() - 1);

				mesh->in_ordered_edges_.resize(meshin.edges.nb(), 2);

				for (int e = 0; e < (int)meshin.edges.nb(); ++e)
				{
					for (int lv = 0; lv < 2; ++lv)
					{
						mesh->in_ordered_edges_(e, lv) = meshin.edges.vertex(e, lv);
					}
				}
				assert(mesh->in_ordered_edges_.size() > 0);

				mesh->in_ordered_faces_.resize(meshin.facets.nb(), meshin.facets.nb_vertices(0));

				for (int f = 0; f < (int)meshin.edges.nb(); ++f)
				{
					assert(mesh->in_ordered_faces_.cols() == meshin.facets.nb_vertices(f));

					for (int lv = 0; lv < mesh->in_ordered_faces_.cols(); ++lv)
					{
						mesh->in_ordered_faces_(f, lv) = meshin.facets.vertex(f, lv);
					}
				}
				assert(mesh->in_ordered_faces_.size() > 0);

				return mesh;
			}
		}

		logger().error("Failed to load mesh");
		return nullptr;
	}

	std::unique_ptr<Mesh> Mesh::create(const std::string &path, const bool non_conforming)
	{
		if (!std::filesystem::exists(path))
		{
			logger().error(path.empty() ? "No mesh provided!" : "Mesh file does not exist: {}", path);
			return nullptr;
		}

		std::string lowername = path;
		std::transform(lowername.begin(), lowername.end(), lowername.begin(), ::tolower);

		if (StringUtils::endswith(lowername, ".hybrid"))
		{
			std::unique_ptr<Mesh> mesh = create(3, non_conforming);
			if (mesh->load(path))
			{
				// TODO add in_ordered_vertices_, in_ordered_edges_, in_ordered_faces_
				return mesh;
			}
		}
		else if (StringUtils::endswith(lowername, ".msh"))
		{
			Eigen::MatrixXd vertices;
			Eigen::MatrixXi cells;
			std::vector<std::vector<int>> elements;
			std::vector<std::vector<double>> weights;
			std::vector<int> body_ids;

			if (!MshReader::load(path, vertices, cells, elements, weights, body_ids))
			{
				logger().error("Failed to load MSH mesh: {}", path);
				return nullptr;
			}

			const int dim = vertices.cols();
			std::unique_ptr<Mesh> mesh = create(vertices, cells, non_conforming);

			// Only tris and tets
			if ((dim == 2 && cells.cols() == 3) || (dim == 3 && cells.cols() == 4))
			{
				mesh->attach_higher_order_nodes(vertices, elements);
				mesh->set_cell_weights(weights);
				// TODO: not clear?
			}

			for (const auto &w : weights)
			{
				if (!w.empty())
				{
					mesh->set_is_rational(true);
					break;
				}
			}

			mesh->set_body_ids(body_ids);

			return mesh;
		}
		else
		{
			GEO::Mesh tmp;
			if (GEO::mesh_load(path, tmp))
			{
				return create(tmp, non_conforming);
			}
		}
		logger().error("Failed to load mesh: {}", path);
		return nullptr;
	}

	std::unique_ptr<Mesh> Mesh::create(
		const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &cells, const bool non_conforming)
	{
		const int dim = vertices.cols();

		std::unique_ptr<Mesh> mesh = create(dim, non_conforming);

		mesh->build_from_matrices(vertices, cells);

		mesh->in_ordered_vertices_ = Eigen::VectorXi::LinSpaced(vertices.rows(), 0, vertices.rows() - 1);
		assert(mesh->in_ordered_vertices_[0] == 0);
		assert(mesh->in_ordered_vertices_[1] == 1);
		assert(mesh->in_ordered_vertices_[2] == 2);
		assert(mesh->in_ordered_vertices_[mesh->in_ordered_vertices_.size() - 1] == vertices.rows() - 1);

		if (dim == 2)
		{
			std::unordered_set<std::pair<int, int>, HashPair> edges;
			for (int f = 0; f < cells.rows(); ++f)
			{
				for (int lv = 0; lv < cells.cols(); ++lv)
				{
					const int v0 = cells(f, lv);
					const int v1 = cells(f, (lv + 1) % cells.cols());
					edges.emplace(std::pair<int, int>(std::min(v0, v1), std::max(v0, v1)));
				}
			}
			mesh->in_ordered_edges_.resize(edges.size(), 2);
			int index = 0;
			for (auto it = edges.begin(); it != edges.end(); ++it)
			{
				mesh->in_ordered_edges_(index, 0) = it->first;
				mesh->in_ordered_edges_(index, 1) = it->second;
				++index;
			}

			assert(mesh->in_ordered_edges_.size() > 0);

			mesh->in_ordered_faces_.resize(0, 0);
		}
		else
		{
			if (cells.cols() == 4)
			{
				get_faces(cells, mesh->in_ordered_faces_);
				igl::edges(mesh->in_ordered_faces_, mesh->in_ordered_edges_);
			}
			// else TODO
		}

		return mesh;
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

	void Mesh::load_boundary_ids(const std::string &path)
	{
		boundary_ids_.resize(n_boundary_elements());

		std::ifstream file(path);

		std::string line;
		int bindex = 0;
		while (std::getline(file, line))
		{
			std::istringstream iss(line);
			int v;
			iss >> v;
			boundary_ids_[bindex] = v;

			++bindex;
		}

		assert(boundary_ids_.size() == size_t(bindex));

		file.close();
	}

	bool Mesh::is_simplex(const int el_id) const
	{
		return elements_tag_[el_id] == ElementType::SIMPLEX;
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