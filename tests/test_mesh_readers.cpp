#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/io/ResourceIO.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <string>

using namespace polyfem;
using namespace polyfem::mesh;

namespace
{
	/// Writes files into a fresh temporary directory, removed on destruction.
	class TempFiles
	{
	public:
		explicit TempFiles(const std::string &name)
			: dir_(std::filesystem::temp_directory_path() / name)
		{
			std::filesystem::remove_all(dir_);
			std::filesystem::create_directories(dir_);
		}
		~TempFiles() { std::filesystem::remove_all(dir_); }

		void write(const std::string &file, const std::string &content) const { std::ofstream(dir_ / file) << content; }
		io::FileSystemIO resources() const { return io::FileSystemIO(dir_); }

	private:
		std::filesystem::path dir_;
	};

	const std::string cube_obj =
		"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n"
		"f 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\nf 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n";
} // namespace

TEST_CASE("OBJ polygon soups in 3D are read as one polyhedron", "[mesh_readers]")
{
	const TempFiles files("polyfem-obj-polyhedron");
	files.write("cube.obj", cube_obj);
	// A free vertex after the faces is the kernel of the polyhedron.
	files.write("cube_kernel.obj", cube_obj + "v 0.25 0.5 0.5\n");

	const auto resources = files.resources();
	const auto cube = MeshLoader(resources).load_fem("cube.obj");
	REQUIRE(cube->is_volume());
	CHECK(cube->n_cells() == 1);
	CHECK(cube->n_faces() == 6);
	CHECK(cube->n_edges() == 12);
	// Without a free vertex, the kernel is the barycenter of the vertices.
	CHECK((dynamic_cast<const Mesh3D &>(*cube).kernel(0) - RowVectorNd::Constant(3, 0.5)).norm() < 1e-14);

	const auto with_kernel = MeshLoader(resources).load_fem("cube_kernel.obj");
	RowVectorNd kernel(3);
	kernel << 0.25, 0.5, 0.5;
	CHECK((dynamic_cast<const Mesh3D &>(*with_kernel).kernel(0) - kernel).norm() < 1e-14);
}

TEST_CASE("OBJ reader handles continuations and blank statements", "[mesh_readers]")
{
	const TempFiles files("polyfem-obj-syntax");
	// A continued face, a whitespace-only line, a tab-led line and other statements.
	files.write("square.obj", "# comment\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n   \n\tjunk\nvt 0 0\nf 1 2 \\\n3\nf 1 3 4\n");
	const auto resources = files.resources();
	const auto mesh = MeshLoader(resources).load_fem("square.obj");
	REQUIRE_FALSE(mesh->is_volume());
	CHECK(mesh->n_elements() == 2);
	CHECK(mesh->n_vertices() == 4);
}

TEST_CASE("OBJ reader reports malformed input", "[mesh_readers]")
{
	const TempFiles files("polyfem-obj-errors");
	files.write("bad_coordinate.obj", "v 0 0 abc\n");
	files.write("bad_corner.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n");
	files.write("bad_vertex.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n");
	files.write("short_face.obj", "v 0 0 0\nf 1\n");
	files.write("no_faces.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n");
	const auto resources = files.resources();

	const auto expect_error = [&](const std::string &file, const std::string &message) {
		CAPTURE(file);
		REQUIRE_THROWS_WITH(MeshLoader(resources).load_fem(file), Catch::Matchers::ContainsSubstring(message));
	};
	expect_error("bad_coordinate.obj", "invalid coordinate");
	expect_error("bad_corner.obj", "invalid face corner");
	expect_error("bad_vertex.obj", "references an invalid vertex");
	expect_error("short_face.obj", "fewer than two corners");
	expect_error("no_faces.obj", "neither cells nor facets");
}

TEST_CASE("mesh loader rejects unsupported formats", "[mesh_readers]")
{
	const TempFiles files("polyfem-mesh-formats");
	files.write("mesh.stl", "solid nothing\nendsolid nothing\n");
	files.write("surface.xyz", "not a mesh\n");
	const auto resources = files.resources();

	REQUIRE_THROWS_WITH(MeshLoader(resources).load_fem("mesh.stl"), Catch::Matchers::ContainsSubstring("Unsupported mesh format"));
	REQUIRE_THROWS_WITH(MeshLoader(resources).load_surface("surface.xyz"), Catch::Matchers::ContainsSubstring("Unable to decode surface"));
}
