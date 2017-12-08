// A dummy GLUP application

#include "navigation.hpp"
#include <geogram/basic/file_system.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram_gfx/glup_viewer/glup_viewer.h>
#include <geogram_gfx/glup_viewer/glup_viewer_gui.h>
#include <geogram_gfx/mesh/mesh_gfx.h>

namespace {
	using namespace std;
	using namespace GEO;
	using namespace poly_fem;

	Navigation::Key key_;

	class DemoGlupApplication : public SimpleMeshApplication {
	public:
		DemoGlupApplication(int argc, char** argv) :
			SimpleMeshApplication(argc, argv, "<filename>")
		{
			name_ = "[Float] Navigation";
		}

		virtual void init_graphics() override {
			SimpleMeshApplication::init_graphics();
			glup_viewer_disable(GLUP_VIEWER_BACKGROUND);
			glup_viewer_disable(GLUP_VIEWER_3D);
			ImGui::GetStyle().WindowRounding = 7.0f;
			ImGui::GetStyle().FrameRounding = 0.0f;
			ImGui::GetStyle().GrabRounding = 0.0f;
		}

		virtual bool load(const std::string &filename) override {
			if(!GEO::FileSystem::is_file(filename)) {
				GEO::Logger::out("I/O") << "is not a file" << std::endl;
				return false;
			}
			SimpleMeshApplication::load(filename);

			// Compute mesh connectivity
			Navigation::prepare_mesh(mesh_);


			// Initialize the key
			key_.fc = mesh_.facets.corner(0, 0);
			key_.v = mesh_.facet_corners.vertex(key_.fc);
			key_.f = 0;
			index_t c2 = mesh_.facets.next_corner_around_facet(key_.f, key_.fc);
			index_t v2 = mesh_.facet_corners.vertex(c2);
			auto minmax = [] (int a, int b) {
				return std::make_pair(std::min(a, b), std::max(a, b));
			};
			auto e0 = minmax(key_.v, v2);
			for (int e = 0; mesh_.edges.nb(); ++e) {
				auto e1 = minmax(mesh_.edges.vertex(e, 0), mesh_.edges.vertex(e, 1));
				if (e0 == e1) {
					key_.e = e;
					break;
				}
			}

			return true;
		}


		virtual void draw_viewer_properties() override {
			ImGui::InputInt("Vtx", &key_.v);
			ImGui::InputInt("Edg", &key_.e);
			ImGui::InputInt("Fct", &key_.f);
			key_.v = std::max(0, std::min((int) mesh_.vertices.nb(), key_.v));
			key_.e = std::max(0, std::min((int) mesh_.edges.nb(), key_.e));
			key_.f = std::max(0, std::min((int) mesh_.facets.nb(), key_.f));
			if (ImGui::Button("Switch Vertex", ImVec2(-1, 0))) {
				key_ = Navigation::switch_vertex(mesh_, key_);
			}
			if (ImGui::Button("Switch Edge", ImVec2(-1, 0))) {
				key_ = Navigation::switch_edge(mesh_, key_);
			}
			if (ImGui::Button("Switch Face", ImVec2(-1, 0))) {

			}
		}

		virtual void draw_scene() override {
			if (mesh()) {
				draw_selected();
			}
			SimpleMeshApplication::draw_scene();
		}

		static vec3 mesh_vertex(const Mesh &M, int i) {
			if (M.vertices.single_precision()) {
				const float *p = M.vertices.single_precision_point_ptr(i);
				return vec3(p[0], p[1], p[2]);
			} else {
				return M.vertices.point(i);
			}
		}

		virtual void draw_selected() {
			glupSetPointSize(GLfloat(10));
			glupEnable(GLUP_VERTEX_COLORS);
			glupColor3f(1.0f, 0.0f, 0.0f);

			// Selected vertex
			glupBegin(GLUP_POINTS);
			glupVertex(mesh_vertex(mesh_, key_.v));
			glupEnd();

			// Selected edge
			glupSetMeshWidth(5);
			glupColor3f(0.7f, 0.0f, 0.0f);
			glupBegin(GLUP_LINES);
			{
				int v0 = mesh_.edges.vertex(key_.e, 0);
				int v1 = mesh_.edges.vertex(key_.e, 1);
				glupVertex(mesh_vertex(mesh_, v0));
				glupVertex(mesh_vertex(mesh_, v1));
			}
			glupEnd();

			// Selected facet
			glupSetMeshWidth(0);
			glupColor3f(1.0f, 0.0f, 0.0f);
			glupBegin(GLUP_TRIANGLES);
			for (int lv = 1; lv + 1 < (int) mesh_.facets.nb_vertices(key_.f); ++lv) {
				int v0 = mesh_.facets.vertex(key_.f, 0);
				int v1 = mesh_.facets.vertex(key_.f, lv);
				int v2 = mesh_.facets.vertex(key_.f, lv+1);
				glupVertex(mesh_vertex(mesh_, v0));
				glupVertex(mesh_vertex(mesh_, v1));
				glupVertex(mesh_vertex(mesh_, v2));
			}
			glupEnd();

			glupDisable(GLUP_VERTEX_COLORS);
		}

	};
}

int main(int argc, char** argv) {
	#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLERS", "1", 1);
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
