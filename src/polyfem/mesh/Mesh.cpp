////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>

#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MshReader.hpp>

#include <polyfem/utils/Logger.hpp>

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
			assert(T.rows() == 4);
			assert(T.cols() >= 1);

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

	std::unique_ptr<Mesh> Mesh::create(GEO::Mesh &meshin, const bool non_conforming)
	{
		if (is_planar(meshin))
		{
			std::unique_ptr<Mesh> mesh;
			if (non_conforming)
				mesh = std::make_unique<NCMesh2D>();
			else
				mesh = std::make_unique<CMesh2D>();
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

				mesh->in_ordered_faces_.resize(0, 0);

				return mesh;
			}
		}
		else
		{
			std::unique_ptr<Mesh> mesh;
			if (non_conforming)
				mesh = std::make_unique<NCMesh3D>();
			else
				mesh = std::make_unique<CMesh3D>();
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
			std::unique_ptr<Mesh> mesh;
			if (non_conforming)
				mesh = std::make_unique<NCMesh3D>();
			else
				mesh = std::make_unique<CMesh3D>();
			if (mesh->load(path))
			{
				//TODO add in_ordered_vertices_, in_ordered_edges_, in_ordered_faces_
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
				return nullptr;

			std::unique_ptr<Mesh> mesh;
			if (vertices.cols() == 2)
				if (non_conforming)
					mesh = std::make_unique<NCMesh2D>();
				else
					mesh = std::make_unique<CMesh2D>();
			else if (non_conforming)
				mesh = std::make_unique<NCMesh3D>();
			else
				mesh = std::make_unique<CMesh3D>();

			mesh->build_from_matrices(vertices, cells);
			// Only tris and tets
			if ((vertices.cols() == 2 && cells.cols() == 3) || (vertices.cols() == 3 && cells.cols() == 4))
			{
				mesh->attach_higher_order_nodes(vertices, elements);
				mesh->cell_weights_ = weights;

				// TODO, not clear?
			}

			for (const auto &w : weights)
			{
				if (!w.empty())
				{
					mesh->is_rational_ = true;
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

		std::unique_ptr<Mesh> mesh;
		if (dim == 2)
		{
			if (non_conforming)
				mesh = std::make_unique<NCMesh2D>();
			else
				mesh = std::make_unique<CMesh2D>();
		}
		else
		{
			assert(dim == 3);
			if (non_conforming)
				mesh = std::make_unique<NCMesh3D>();
			else
				mesh = std::make_unique<CMesh3D>();
		}

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
			if (cells.rows() == 4)
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
			return elements_tag_[el_id] == ElementType::RegularInteriorCube
				   || elements_tag_[el_id] == ElementType::RegularBoundaryCube;
			// || elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube
			// || elements_tag_[el_id] == ElementType::SimpleSingularBoundaryCube;
		}
		else
		{
			return elements_tag_[el_id] == ElementType::RegularInteriorCube
				   || elements_tag_[el_id] == ElementType::RegularBoundaryCube;
			// || elements_tag_[el_id] == ElementType::InterfaceCube
			// || elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube;
		}
	}

	// -----------------------------------------------------------------------------

	bool Mesh::is_cube(const int el_id) const
	{
		return elements_tag_[el_id] == ElementType::InterfaceCube
			   || elements_tag_[el_id] == ElementType::RegularInteriorCube
			   || elements_tag_[el_id] == ElementType::RegularBoundaryCube
			   || elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube
			   || elements_tag_[el_id] == ElementType::SimpleSingularBoundaryCube
			   || elements_tag_[el_id] == ElementType::MultiSingularInteriorCube
			   || elements_tag_[el_id] == ElementType::MultiSingularBoundaryCube;
	}

	// -----------------------------------------------------------------------------

	bool Mesh::is_polytope(const int el_id) const
	{
		return elements_tag_[el_id] == ElementType::InteriorPolytope
			   || elements_tag_[el_id] == ElementType::BoundaryPolytope;
	}

	void Mesh::load_boundary_ids(const std::string &path)
	{
		boundary_ids_.resize(is_volume() ? n_faces() : n_edges());

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
		return elements_tag_[el_id] == ElementType::Simplex;
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
} // namespace polyfem::mesh