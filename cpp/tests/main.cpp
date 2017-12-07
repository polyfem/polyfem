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

	int v_;
	int e_;
	int f_;

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

		virtual void draw_viewer_properties() override {
			ImGui::InputInt("Vtx", &v_);
			ImGui::InputInt("Edg", &e_);
			ImGui::InputInt("Fct", &f_);
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
			glupVertex(mesh_vertex(mesh_, v_));
			glupEnd();

			// Selected edge
			// glupBegin(GLUP_LINES);
			// glupVertex(mesh_vertex(mesh_, v_));
			// glupVertex(mesh_vertex(mesh_, v_));
			// glupEnd();

			// Selected facet
			glupSetMeshWidth(0);
			glupBegin(GLUP_TRIANGLES);
			for (int lv = 1; lv + 1 < (int) mesh_.facets.nb_vertices(f_); ++lv) {
				int v0 = mesh_.facets.vertex(f_, 0);
				int v1 = mesh_.facets.vertex(f_, lv);
				int v2 = mesh_.facets.vertex(f_, lv+1);
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
