#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <h5pp/h5pp.h>

#include <polyfem/io/Checkpoint.hpp>
#include <polyfem/io/InputLoader.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/mesh/MeshLoader.hpp>

#include <filesystem>
#include <fstream>

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
		file.writeDataset(std::string(R"({"geometry":[]})"), "/config");
		file.writeDataset(std::string("embedded text"), "/assets/note.txt");
		Eigen::MatrixXd vertices(3, 2);
		vertices << 0, 0, 1, 0, 0, 1;
		Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> cells(1, 3);
		cells << 0, 1, 2;
		file.writeDataset(vertices, "/meshes/triangle/vertices");
		file.writeDataset(cells, "/meshes/triangle/cells");
		file.writeAttribute(long(polyfem::mesh::MESH_SCHEMA_VERSION), "/meshes/triangle", "schema_version");
		file.writeAttribute(long(2), "/meshes/triangle", "dimension");
		file.writeAttribute(std::string("fem"), "/meshes/triangle", "mesh_type");
	}
	const io::LoadedInput loaded = io::load_hdf5_input(bundle);
	CHECK(loaded.config["geometry"].empty());
	CHECK(loaded.resources->read_string("assets/note.txt") == "embedded text");
	CHECK(loaded.resources->materialize("assets/note.txt").extension() == ".txt");
	mesh::MeshLoader loader(*loaded.resources);
	const auto mesh = loader.load_fem("meshes/triangle");
	REQUIRE(mesh != nullptr);
	CHECK(mesh->dimension() == 2);
	CHECK(mesh->n_vertices() == 3);
	CHECK(mesh->n_elements() == 1);
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
	auto mesh = mesh::Mesh::create(vertices, cells, false);
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
