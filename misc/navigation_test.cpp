// A dummy GLUP application

#include <polyfem/Navigation.hpp>
#include <polyfem/MeshUtils.hpp>
#include <polyfem/PolygonUtils.hpp>
#include <polyfem/Singularities.hpp>
#include <polyfem/Refinement.hpp>
#include <geogram/basic/file_system.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_preprocessing.h>
#include <geogram_gfx/glup_viewer/glup_viewer.h>
#include <geogram_gfx/glup_viewer/glup_viewer_gui.h>
#include <geogram_gfx/mesh/mesh_gfx.h>
#include <Eigen/Dense>

namespace {
	using namespace std;
	using namespace GEO;
	using namespace polyfem;

	Navigation::Index idx_;

	Eigen::VectorXi singular_vertices_;
	Eigen::MatrixX2i singular_edges_;
    
	float t_;

	bool show_index_;

	class DemoGlupApplication : public SimpleMeshApplication {
	public:
		DemoGlupApplication(int argc, char** argv) :
			SimpleMeshApplication(argc, argv, "<filename>")
		{
			name_ = "[Float] Navigation";
			t_ = 0.5;
			show_index_ = true;
		}

		virtual void init_graphics() override {
			SimpleMeshApplication::init_graphics();
			glup_viewer_disable(GLUP_VIEWER_BACKGROUND);
			glup_viewer_disable(GLUP_VIEWER_3D);
			ImGui::GetStyle().WindowRounding = 7.0f;
			ImGui::GetStyle().FrameRounding = 0.0f;
			ImGui::GetStyle().GrabRounding = 0.0f;
			retina_mode_ = false;
			scaling_ = 1.0f;
		}

		virtual bool load(const std::string &filename) override {
			if(!GEO::FileSystem::is_file(filename)) {
				GEO::Logger::out("I/O") << "is not a file" << std::endl;
				return false;
			}
			SimpleMeshApplication::load(filename);
			mesh_.vertices.set_double_precision();
			polyfem::orient_normals_2d(mesh_);
			mesh_.vertices.set_single_precision();

			// Compute mesh connectivity
			Navigation::prepare_mesh(mesh_);
			GEO::Attribute<GEO::index_t> c2e(mesh_.facet_corners.attributes(), "edge_id");


			// Initialize the key
			idx_ = Navigation::get_index_from_face(mesh_, c2e, 0, 0);

			// Compute singularities
			polyfem::singularity_graph(mesh_, singular_vertices_, singular_edges_);

			// Compute element types
			compute_types();

			return true;
		}

		void compute_types() {
			std::vector<ElementType> tags;
			polyfem::compute_element_tags(mesh_, tags);
			GEO::Attribute<int> attrs(mesh_.facets.attributes(), "tags");
			for (index_t f = 0; f < mesh_.facets.nb(); ++f) {
				switch (tags[f]) {
				case ElementType::RegularInteriorCube:        attrs[f] = 0; break;
				case ElementType::SimpleSingularInteriorCube: attrs[f] = 1; break;
				case ElementType::MultiSingularInteriorCube:  attrs[f] = 2; break;
				case ElementType::RegularBoundaryCube:        attrs[f] = 3; break;
				case ElementType::SimpleSingularBoundaryCube: attrs[f] = 4; break;
				case ElementType::MultiSingularBoundaryCube:  attrs[f] = 5; break;
				case ElementType::InteriorPolytope:           attrs[f] = 6; break;
				case ElementType::BoundaryPolytope:           attrs[f] = 7; break;
				case ElementType::Undefined:                  attrs[f] = 8; break;
				default: attrs[f] = -1; break;
				}
				// std::cout << attrs[f] << std::endl;
			}
		}

		void compute_visibility_kernel() {
			index_t f = idx_.face;
			if (f < mesh_.facets.nb()) {
				Eigen::MatrixXd IV(mesh_.facets.nb_vertices(f), 2);
				for (index_t lv = 0; lv < mesh_.facets.nb_vertices(f); ++lv) {
					index_t v = mesh_.facets.vertex(f, lv);
					vec3 p = mesh_vertex(mesh_, v);
					// std::cout << p << std::endl;
					IV.row(lv) << p[0], p[1];
				}
				Eigen::MatrixXd OV;
				polyfem::compute_visibility_kernel(IV, OV);
				mesh_.clear();
				GEO::vector<index_t> poly;
				// std::cout << OV << std::endl;
				for (index_t v = 0; v < OV.rows(); ++v) {
					index_t vv = mesh_.vertices.create_vertex();
					assert(v == vv);
					float *p = mesh_.vertices.single_precision_point_ptr(v);
					p[0] = (float) OV(v, 0);
					p[1] = (float) OV(v, 1);
					poly.push_back(v);
					// std::cout << v << ' ' << p[0] << ' ' << p[1] << std::endl;
				}
				mesh_.facets.create_polygon(poly);
			}
		}

		virtual void draw_viewer_properties() override {
			ImGui::Text("Vertex: %d", idx_.vertex);
			ImGui::Text("Edge:   %d", idx_.edge);
			ImGui::Text("Facet:  %d", idx_.face);
			idx_.vertex = std::max(0, std::min((int) mesh_.vertices.nb(), idx_.vertex));
			idx_.edge = std::max(0, std::min((int) mesh_.edges.nb(), idx_.edge));
			idx_.face = std::max(0, std::min((int) mesh_.facets.nb(), idx_.face));
			GEO::Attribute<GEO::index_t> c2e(mesh_.facet_corners.attributes(), "edge_id");

			ImGui::Checkbox("Show Index", &show_index_);
			if (ImGui::Button("Switch Vertex", ImVec2(-1, 0))) {
				idx_ = Navigation::switch_vertex(mesh_, idx_);
			}
			if (ImGui::Button("Switch Edge", ImVec2(-1, 0))) {
				idx_ = Navigation::switch_edge(mesh_, c2e, idx_);
			}
			if (ImGui::Button("Switch Face", ImVec2(-1, 0))) {
				auto tmp = Navigation::switch_face(mesh_, c2e, idx_);
				if (tmp.face != -1) {
					idx_ = tmp;
				}
			}

			ImGui::Separator();

			if (ImGui::Button("Kill Singularities", ImVec2(-1, 0))) {
				polyfem::create_patch_around_singularities(mesh_, singular_vertices_, singular_edges_);
				Navigation::prepare_mesh(mesh_);
				polyfem::singularity_graph(mesh_, singular_vertices_, singular_edges_);
				compute_types();
				GEO::mesh_save(mesh_, "foo.obj");
			}

			ImGui::DragFloat("t", &t_, 0.01f, 0.0f, 1.0f);
			static bool split_polygons = false;
			ImGui::Checkbox("Split Polygons", &split_polygons);
			static bool force_quad_ring = false;
			ImGui::Checkbox("Force quad ring", &force_quad_ring);
			if (ImGui::Button("Refine", ImVec2(-1, 0))) {
				GEO::Mesh tmp;
				if (split_polygons == false) {
					polyfem::refine_polygonal_mesh(mesh_, tmp, polyfem::Polygons::no_split_func());
				} else if (force_quad_ring) {
					polyfem::refine_polygonal_mesh(mesh_, tmp, polyfem::Polygons::catmul_clark_split_func());
				} else {
					polyfem::refine_polygonal_mesh(mesh_, tmp, polyfem::Polygons::polar_split_func(t_));
				}
				mesh_.copy(tmp);
				// std::cout << mesh_.vertices.nb() << std::endl;
				Navigation::prepare_mesh(mesh_);
				polyfem::singularity_graph(mesh_, singular_vertices_, singular_edges_);
				compute_types();
				idx_ = Navigation::get_index_from_face(mesh_, c2e, 0, 0);
				// GEO::mesh_save(mesh_, "foo.obj");
				mesh_gfx_.set_mesh(&mesh_);
			}

			if (ImGui::Button("Visibility Kernel", ImVec2(-1, 0))) {
				compute_visibility_kernel();
				Navigation::prepare_mesh(mesh_);
				polyfem::singularity_graph(mesh_, singular_vertices_, singular_edges_);
				compute_types();
				idx_ = Navigation::get_index_from_face(mesh_, c2e, 0, 0);
			}
		}

		virtual void draw_scene() override {
			if (mesh()) {
				if (show_index_) {
					draw_selected();
				}
				//draw_singular();
			}
			SimpleMeshApplication::draw_scene();
		}

		virtual void draw_selected() {
			glupSetPointSize(GLfloat(10));
			glupEnable(GLUP_VERTEX_COLORS);
			glupColor3f(1.0f, 0.0f, 0.0f);

			// Selected vertex
			glupBegin(GLUP_POINTS);
			glupVertex(mesh_vertex(mesh_, idx_.vertex));
			glupEnd();

			// Selected edge
			glupSetMeshWidth(5);
			glupColor3f(0.0f, 0.8f, 0.0f);
			glupBegin(GLUP_LINES);
			{
				int v0 = mesh_.edges.vertex(idx_.edge, 0);
				int v1 = mesh_.edges.vertex(idx_.edge, 1);
				glupVertex(mesh_vertex(mesh_, v0));
				glupVertex(mesh_vertex(mesh_, v1));
			}
			// Boundary edges
			GEO::Attribute<bool> boundary(mesh_.edges.attributes(), "boundary_edge");
			glupColor3f(0.7f, 0.7f, 0.0f);
			for (int e = 0; e < (int) mesh_.edges.nb(); ++e) {
				if (boundary[e]) {
					int v0 = mesh_.edges.vertex(e, 0);
					int v1 = mesh_.edges.vertex(e, 1);
					// glupVertex(mesh_vertex(mesh_, v0));
					// glupVertex(mesh_vertex(mesh_, v1));
				}
			}
			glupEnd();

			// Selected facet
			glupSetMeshWidth(0);
			glupColor3f(1.0f, 0.0f, 0.0f);
			glupBegin(GLUP_TRIANGLES);
			for (int lv = 1; lv + 1 < (int) mesh_.facets.nb_vertices(idx_.face); ++lv) {
				int v0 = mesh_.facets.vertex(idx_.face, 0);
				int v1 = mesh_.facets.vertex(idx_.face, lv);
				int v2 = mesh_.facets.vertex(idx_.face, lv+1);
				glupVertex(mesh_vertex(mesh_, v0));
				glupVertex(mesh_vertex(mesh_, v1));
				glupVertex(mesh_vertex(mesh_, v2));
			}
			glupEnd();

			glupDisable(GLUP_VERTEX_COLORS);
		}

		void draw_singular() {
			glupSetPointSize(GLfloat(10));
			glupEnable(GLUP_VERTEX_COLORS);
			glupColor3f(0.0f, 0.0f, 0.8f);

			// Selected vertex
			glupBegin(GLUP_POINTS);
			for (int i = 0; i < singular_vertices_.size(); ++i) {
				glupVertex(mesh_vertex(mesh_, singular_vertices_[i]));
			}
			glupEnd();

			// Selected edge
			glupSetMeshWidth(5);
			glupBegin(GLUP_LINES);
			for (int e = 0; e < singular_edges_.rows(); ++e) {
				int v0 = singular_edges_(e, 0);
				int v1 = singular_edges_(e, 1);
				glupVertex(mesh_vertex(mesh_, v0));
				glupVertex(mesh_vertex(mesh_, v1));
			}
			glupEnd();
			glupDisable(GLUP_VERTEX_COLORS);
		}

	};
}

int main(int argc, char** argv) {
	#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
	#endif
	GEO::initialize();
	GEO::CmdLine::import_arg_group("standard");
	GEO::CmdLine::import_arg_group("algo");
	GEO::CmdLine::import_arg_group("gfx");
	DemoGlupApplication app(argc, argv);
	GEO::CmdLine::set_arg("gfx:geometry", "1024x1024");
	app.start();
	return 0;
}
