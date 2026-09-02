#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <h5pp/h5pp.h>

#include <polyfem/io/Checkpoint.hpp>
#include <polyfem/io/InputLoader.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

TEST_CASE("HDF5", "[hdf5]")
{
	using MatrixXl = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

	const std::string hdf5_file = std::string(POLYFEM_DATA_DIR) + "/test.hdf5";
	h5pp::File file(hdf5_file, h5pp::FileAccess::READONLY);
	std::string json_string = file.readDataset<std::string>("json");

	nlohmann::json in_args = nlohmann::json::parse(json_string);
	in_args["root_path"] = hdf5_file;

	std::vector<std::string> names = file.findGroups("", "/meshes");
	CHECK(names.size() == 2);
	CHECK(names[0] == "hdf5_0");
	CHECK(names[1] == "hdf5_1");
	std::vector<Eigen::MatrixXi> cells(names.size());
	std::vector<Eigen::MatrixXd> vertices(names.size());

	for (int i = 0; i < names.size(); ++i)
	{
		const std::string &name = names[i];
		cells[i] = file.readDataset<MatrixXl>("/meshes/" + name + "/c").cast<int>();
		vertices[i] = file.readDataset<Eigen::MatrixXd>("/meshes/" + name + "/v");
	}

	polyfem::io::HDF5IO resources(hdf5_file);
	CHECK_THROWS(polyfem::mesh::MeshLoader(resources).load_fem("meshes/hdf5_0"));
}

TEST_CASE("ResourceIO filesystem and HDF5 backends", "[hdf5][resource_io]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path directory = fs::temp_directory_path() / "polyfem-resource-io-test";
	fs::remove_all(directory);
	fs::create_directories(directory / "nested");
	{
		std::ofstream out(directory / "nested" / "resource.txt");
		out << "resource contents";
	}
	io::FileSystemIO filesystem(directory);
	CHECK(filesystem.read_string("nested/resource.txt") == "resource contents");
	CHECK(filesystem.with_root("nested")->exists("resource.txt"));
	CHECK(filesystem.glob("nested/*.txt") == std::vector<std::string>{"nested/resource.txt"});

	const fs::path bundle = directory / "bundle.h5";
	{
		h5pp::File file(bundle.string(), h5pp::FileAccess::REPLACE);
		file.writeDataset(std::string(R"({"common":"configs/common.json","geometry":[]})"), "/config");
		file.writeDataset(std::string(R"({"materials":{"type":"NeoHookean"}})"), "/configs/common.json");
		file.writeDataset(std::string("common-local resource"), "/configs/local.txt");
		file.writeDataset(std::string("embedded text"), "/assets/note.txt");
		file.writeDataset(
			std::string(
				"# vtk DataFile Version 2.0\n"
				"fibers\n"
				"ASCII\n"
				"DATASET UNSTRUCTURED_GRID\n"
				"CELL_DATA 2\n"
				"VECTORS FIB_DIR1 double\n"
				"2 0 0\n"
				"0 3 0\n"),
			"/assets/fibers.vtk");
		Eigen::MatrixXd vertices(3, 2);
		vertices << 0, 0, 1, 0, 0, 1;
		Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> cells(1, 3);
		cells << 0, 1, 2;
		Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> boundary_elements(3, 2);
		boundary_elements << 0, 1, 1, 2, 2, 0;
		file.writeDataset(vertices, "/meshes/triangle/vertices");
		file.writeDataset(cells, "/meshes/triangle/cells");
		file.writeDataset(std::vector<int>{7}, "/meshes/triangle/body_ids");
		file.writeDataset(std::vector<int>{9}, "/meshes/triangle/geometry_ids");
		file.writeDataset(boundary_elements, "/meshes/triangle/boundary_elements");
		file.writeDataset(std::vector<int>{11, 12, 13}, "/meshes/triangle/boundary_ids");
		file.writeAttribute(long(polyfem::mesh::MESH_SCHEMA_VERSION), "/meshes/triangle", "schema_version");
		file.writeAttribute(long(2), "/meshes/triangle", "dimension");
		file.writeAttribute(std::string("fem"), "/meshes/triangle", "mesh_type");

		file.writeDataset(vertices, "/surfaces/triangle/vertices");
		file.writeDataset(boundary_elements, "/surfaces/triangle/edges");
		file.writeDataset(cells, "/surfaces/triangle/faces");
		file.writeAttribute(long(polyfem::mesh::MESH_SCHEMA_VERSION), "/surfaces/triangle", "schema_version");
		file.writeAttribute(long(2), "/surfaces/triangle", "dimension");
		file.writeAttribute(std::string("surface"), "/surfaces/triangle", "mesh_type");

		Eigen::MatrixXd poly_vertices(4, 3);
		poly_vertices << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
		Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> poly_cell(1, 4);
		poly_cell << 0, 1, 2, 3;
		Eigen::MatrixXd kernel(1, 3);
		kernel << 0.25, 0.25, 0.25;
		file.writeDataset(poly_vertices, "/meshes/polyhedron/vertices");
		file.writeDataset(poly_cell, "/meshes/polyhedron/cells");
		file.writeDataset(std::vector<int>{0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3}, "/meshes/polyhedron/faces");
		file.writeDataset(std::vector<long>{0, 3, 6, 9, 12}, "/meshes/polyhedron/face_offsets");
		file.writeDataset(std::vector<int>{0, 1, 2, 3}, "/meshes/polyhedron/cell_faces");
		file.writeDataset(std::vector<long>{0, 4}, "/meshes/polyhedron/cell_face_offsets");
		file.writeDataset(std::vector<int>{1, 1, 1, 1}, "/meshes/polyhedron/cell_face_orientations");
		file.writeDataset(std::vector<int>{0}, "/meshes/polyhedron/cell_is_hex");
		file.writeDataset(kernel, "/meshes/polyhedron/cell_kernel_points");
		file.writeAttribute(long(polyfem::mesh::MESH_SCHEMA_VERSION), "/meshes/polyhedron", "schema_version");
		file.writeAttribute(long(3), "/meshes/polyhedron", "dimension");
		file.writeAttribute(std::string("fem"), "/meshes/polyhedron", "mesh_type");
	}
	const io::LoadedInput loaded = io::load_hdf5_input(bundle);
	CHECK(loaded.config["geometry"].empty());
	CHECK(varform::uses_varform_state(loaded.config, *loaded.resources));
	json effective_config = loaded.config;
	auto common_resources = utils::apply_common_params(effective_config, *loaded.resources);
	REQUIRE(common_resources != nullptr);
	CHECK(common_resources->read_string("local.txt") == "common-local resource");
	CHECK(loaded.resources->read_string("assets/note.txt") == "embedded text");
	CHECK(loaded.resources->materialize("assets/note.txt").extension() == ".txt");
	assembler::FiberDirection fibers;
	fibers.resize(3);
	fibers.add_multimaterial(
		0,
		json{{"type", "per_element_file"}, {"path", "assets/fibers.vtk"}, {"field", "FIB_DIR1"}},
		"", *loaded.resources);
	CHECK(fibers(0, 0, 0, 0, 0, 0, 0, 0).isApprox(Eigen::Vector3d::UnitX()));
	CHECK(fibers(0, 0, 0, 0, 0, 0, 0, 1).isApprox(Eigen::Vector3d::UnitY()));
	mesh::MeshLoader loader(*loaded.resources);
	const auto mesh = loader.load_fem("meshes/triangle");
	REQUIRE(mesh != nullptr);
	CHECK(mesh->dimension() == 2);
	CHECK(mesh->n_vertices() == 3);
	CHECK(mesh->n_elements() == 1);
	CHECK(mesh->get_body_id(0) == 7);
	CHECK(mesh->get_geometry_id(0) == 9);
	std::vector<int> boundary_ids;
	for (int edge = 0; edge < mesh->n_edges(); ++edge)
		boundary_ids.push_back(mesh->get_boundary_id(edge));
	std::sort(boundary_ids.begin(), boundary_ids.end());
	CHECK(boundary_ids == std::vector<int>{11, 12, 13});
	const mesh::SurfaceMesh surface = loader.load_surface("surfaces/triangle");
	CHECK(surface.vertices.rows() == 3);
	CHECK(surface.vertices.cols() == 2);
	CHECK(surface.edges.rows() == 3);
	CHECK(surface.faces.rows() == 1);
	const auto polyhedron = loader.load_fem("meshes/polyhedron");
	REQUIRE(polyhedron != nullptr);
	CHECK(polyhedron->n_vertices() == 4);
	CHECK(polyhedron->n_cells() == 1);
	const auto *conforming_polyhedron = dynamic_cast<const mesh::CMesh3D *>(polyhedron.get());
	REQUIRE(conforming_polyhedron != nullptr);
	CHECK(conforming_polyhedron->kernel(0).isApprox(Eigen::RowVector3d(0.25, 0.25, 0.25)));
	fs::remove_all(directory);
}

TEST_CASE("Checkpoint metadata and state round trip", "[hdf5][checkpoint]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path path = fs::temp_directory_path() / "polyfem-checkpoint-test.h5";
	fs::remove(path);
	Eigen::MatrixXd vertices(3, 2);
	vertices << 0, 0, 1, 0, 0, 1;
	Eigen::MatrixXi cells(1, 3);
	cells << 0, 1, 2;
	auto mesh = mesh::Mesh::create(mesh::MeshData(vertices, cells), false);
	REQUIRE(mesh != nullptr);
	io::CheckpointMetadata metadata;
	metadata.formulation = "Laplacian";
	metadata.step = 3;
	metadata.time = 0.3;
	metadata.dt = 0.1;
	metadata.remaining_steps = 2;
	metadata.output_index = 3;
	{
		io::CheckpointWriter writer(path, json{{"time", {{"dt", 0.1}}}}, metadata);
		writer.write_mesh("/checkpoint/meshes/active", *mesh);
		writer.write_matrix("/checkpoint/state/solution", Eigen::MatrixXd::Ones(3, 1));
		writer.finalize();
	}
	io::CheckpointReader reader(path);
	CHECK(reader.metadata().schema_version == io::CHECKPOINT_SCHEMA_VERSION);
	CHECK(reader.metadata().formulation == "Laplacian");
	CHECK(reader.metadata().step == 3);
	CHECK(reader.read_matrix("/checkpoint/state/solution").isOnes());
	fs::remove(path);
}
