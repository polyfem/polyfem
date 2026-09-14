#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <h5pp/h5pp.h>

#include <polyfem/io/Checkpoint.hpp>
#include <polyfem/io/InputLoader.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/State.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace
{
	void write_test_checkpoint_state(polyfem::io::CheckpointWriter &writer)
	{
		Eigen::MatrixXd vertices(3, 2);
		vertices << 0, 0, 1, 0, 0, 1;
		Eigen::MatrixXi cells(1, 3);
		cells << 0, 1, 2;
		writer.write_mesh("/checkpoint/meshes/active", polyfem::mesh::MeshData(vertices, cells));
		writer.write_matrix("/checkpoint/state/solution", Eigen::MatrixXd::Ones(3, 1));
	}
} // namespace

TEST_CASE("Resource numeric reads distinguish file bytes and typed data", "[hdf5][resource_io]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path directory = fs::temp_directory_path() / "polyfem-resource-numeric-types";
	fs::create_directories(directory);
	const std::string text = "1 2\n3 4\n";
	std::ofstream(directory / "values.txt") << text;
	std::ofstream(directory / "long.txt") << "4294967296\n";
	std::ofstream(directory / "invalid.txt") << "1 nope\n";
	std::ofstream(directory / "invalid.bin", std::ios::binary) << "short";
	std::ofstream(directory / "huge.txt") << "18446744073709551616\n";
	Eigen::MatrixXi integers(2, 2);
	integers << 1, 2, 3, 4;
	const Eigen::MatrixXd doubles = integers.cast<double>();
	REQUIRE(io::write_matrix((directory / "values.bin").string(), doubles));
	const fs::path bundle = directory / "input.h5";
	{
		h5pp::File file(bundle.string(), h5pp::FileAccess::REPLACE);
		file.writeDataset(std::vector<unsigned char>(text.begin(), text.end()), "/values.txt");
		const std::string binary = io::FileSystemIO(directory).read_string("values.bin");
		file.writeDataset(std::vector<unsigned char>(binary.begin(), binary.end()), "/values.bin");
		file.writeDataset(text, "/string.txt");
		file.writeDataset(std::string("4294967296\n"), "/long.txt");
		file.writeDataset(std::string("1 nope\n"), "/invalid.txt");
		file.writeDataset(std::string("short"), "/invalid.bin");
		file.writeDataset(std::string("18446744073709551616\n"), "/huge.txt");
		file.writeDataset(integers, "/int32");
		file.writeDataset(integers.cast<int64_t>(), "/int64");
		file.writeDataset(integers.cast<float>() * 0.5f, "/float32");
		file.writeDataset(std::vector<int64_t>{4294967296LL}, "/wide");
		file.writeDataset(std::vector<int64_t>{(int64_t(1) << 54) + 1}, "/exact");
		file.writeDataset(std::vector<uint64_t>{std::numeric_limits<uint64_t>::max()}, "/unsigned_overflow");
		file.writeDataset(std::vector<int64_t>{-1, 2}, "/signed");
		file.writeDataset(std::vector<uint16_t>{1, 65535}, "/uint16");
		file.writeDataset(std::vector<int>{}, "/empty");
		file.writeDataset(std::vector<int>{1, 2, 3, 4}, "/rank3", {1, 2, 2});
	}
	{
		io::FileSystemIO filesystem(directory);
		io::HDF5IO hdf5(bundle);
		for (const io::ResourceIO &resources : {std::cref<const io::ResourceIO>(filesystem), std::cref<const io::ResourceIO>(hdf5)})
		{
			CAPTURE(resources.describe("values.txt"));
			CHECK(resources.read_int_matrix("values.txt") == integers);
			CHECK(resources.read_matrix("values.txt") == integers.cast<double>());
			CHECK(resources.read_int_vector("values.txt") == std::vector<int>{1, 3, 2, 4});
			CHECK(resources.read_double_vector("values.txt") == std::vector<double>{1, 3, 2, 4});
			CHECK(resources.read_long_vector("values.txt") == std::vector<long>{1, 3, 2, 4});
			CHECK(resources.read_matrix("values.bin") == doubles);
			CHECK(resources.read_double_vector("values.bin") == std::vector<double>{1, 3, 2, 4});
			if (sizeof(long) > sizeof(int))
				CHECK(resources.read_long_vector("long.txt") == std::vector<long>{static_cast<long>(4294967296LL)});
			else
				CHECK_THROWS(resources.read_long_vector("long.txt"));
			CHECK_THROWS(resources.read_int_vector("long.txt"));
			CHECK_THROWS(resources.read_int_matrix("invalid.txt"));
			CHECK_THROWS(resources.read_matrix("invalid.txt"));
			CHECK_THROWS(resources.read_matrix("invalid.bin"));
			CHECK_THROWS(resources.read_long_vector("huge.txt"));
		}
		CHECK(hdf5.read_int_vector("string.txt") == std::vector<int>{1, 3, 2, 4});
		CHECK(hdf5.read_int_matrix("int32") == integers);
		CHECK(hdf5.read_int_matrix("int64") == integers);
		CHECK(hdf5.read_matrix("int32") == integers.cast<double>());
		CHECK(hdf5.read_matrix("float32") == integers.cast<double>() * 0.5);
		CHECK(hdf5.read_int_vector("int64") == std::vector<int>{1, 3, 2, 4});
		CHECK(hdf5.read_int_vector("signed") == std::vector<int>{-1, 2});
		CHECK(hdf5.read_int_vector("uint16") == std::vector<int>{1, 65535});
		CHECK(hdf5.read_int_vector("empty").empty());
		CHECK_THROWS(hdf5.read_int_vector("wide"));
		CHECK_THROWS(hdf5.read_int_matrix("float32"));
		CHECK_THROWS(hdf5.read_long_vector("unsigned_overflow"));
		CHECK_THROWS(hdf5.read_string("int32"));
		CHECK_THROWS(hdf5.materialize("int32"));
		CHECK_THROWS(hdf5.read_matrix("rank3"));
		if (sizeof(long) > sizeof(int))
		{
			CHECK(hdf5.read_long_vector("wide") == std::vector<long>{static_cast<long>(4294967296LL)});
			CHECK(hdf5.read_long_vector("exact") == std::vector<long>{static_cast<long>((int64_t(1) << 54) + 1)});
		}
		else
			CHECK_THROWS(hdf5.read_long_vector("wide"));
	}
	fs::remove_all(directory);
}

TEST_CASE("Resource paths and globs use normalized logical names", "[hdf5][resource_io]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path directory = fs::temp_directory_path() / "polyfem-resource-globs";
	fs::create_directories(directory / "nested/deeper");
	const std::vector<std::string> paths{"root.txt", "nested/a.txt", "nested/deeper/b.txt"};
	const fs::path bundle = directory / "input.h5";
	{
		h5pp::File file(bundle.string(), h5pp::FileAccess::REPLACE);
		for (const auto &path : paths)
		{
			std::ofstream(directory / path) << path;
			file.writeDataset(path, "/assets/" + path);
		}
		file.writeDataset(std::string("literal backslash"), "/assets/back\\slash.txt");
		file.writeDataset(std::string("literal colon"), "/assets/C:/value.txt");
	}
	{
		io::FileSystemIO filesystem(directory);
		io::HDF5IO hdf5(bundle, "/assets");
		for (const io::ResourceIO &resources : {std::cref<const io::ResourceIO>(filesystem), std::cref<const io::ResourceIO>(hdf5)})
		{
			CAPTURE(resources.describe(""));
			CHECK(resources.glob("nested/*.txt") == std::vector<std::string>{"nested/a.txt"});
			CHECK(resources.glob("./nested//*.txt") == resources.glob("nested/*.txt"));
			CHECK(resources.glob("nested/../nested/?.txt") == std::vector<std::string>{"nested/a.txt"});
			CHECK(resources.glob("nested/**/*.txt") == std::vector<std::string>{"nested/a.txt", "nested/deeper/b.txt"});
			CHECK(resources.glob("./root.txt") == std::vector<std::string>{"root.txt"});
			CHECK(resources.glob("missing/**/*.txt").empty());
			CHECK(resources.glob("root.txt/*.txt").empty());
			const auto recursive = resources.glob("**/*.txt");
			for (const auto &path : paths)
				CHECK(std::find(recursive.begin(), recursive.end(), path) != recursive.end());
			const auto nested = resources.with_root("nested/./deeper/..");
			const auto parent_files = nested->glob("../*.txt");
			CHECK(std::find(parent_files.begin(), parent_files.end(), "../root.txt") != parent_files.end());
			CHECK(nested->glob("**/a.txt") == std::vector<std::string>{"a.txt"});
		}
		CHECK(hdf5.canonical_path("/assets//nested/./../root.txt") == "/assets/root.txt");
		CHECK(hdf5.canonical_path("../../root.txt") == "/root.txt");
		CHECK(hdf5.canonical_path("C:/value.txt") == "/assets/C:/value.txt");
		CHECK(hdf5.read_string("back\\slash.txt") == "literal backslash");
		CHECK(hdf5.glob("/assets/nested/*.txt") == std::vector<std::string>{"/assets/nested/a.txt"});
		CHECK(hdf5.glob("back\\*.txt") == std::vector<std::string>{"back\\slash.txt"});
		CHECK(hdf5.list("/assets/nested") == std::vector<std::string>{"/assets/nested/a.txt", "/assets/nested/deeper"});
		io::HDF5IO mounted(bundle, "/nested", {}, "/assets");
		CHECK(mounted.read_string("/root.txt") == "root.txt");
		CHECK(mounted.glob("../*.txt") == std::vector<std::string>{"../back\\slash.txt", "../root.txt"});
		CHECK(mounted.resolve("../../root.txt") == "/assets/root.txt");
	}
	fs::remove_all(directory);
}

TEST_CASE("Checkpoint dependencies retain roots and later accesses", "[hdf5][resource_io][checkpoint]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path directory = fs::temp_directory_path() / "polyfem-checkpoint-rooted-resources";
	fs::create_directories(directory / "inputs/a");
	fs::create_directories(directory / "inputs/b");
	std::ofstream(directory / "inputs/a/value.txt") << "first";
	std::ofstream(directory / "inputs/b/value.txt") << "second";
	{
		const io::FileSystemIO resources(directory / "inputs");
		const auto first = resources.with_root("a");
		const auto second = resources.with_root("b");
		CHECK(first->read_string("./value.txt") == "first");
		const auto snapshot = resources.accessed_resources();
		CHECK(snapshot == std::vector<std::string>{resources.canonical_path("a/value.txt")});
		io::CheckpointMetadata metadata;
		metadata.dt = 0.1;
		{
			io::CheckpointWriter writer(directory / "first.h5", json::object(), metadata);
			write_test_checkpoint_state(writer);
			writer.embed_resources(*first);
			writer.finalize();
		}
		CHECK(second->read_string("value.txt") == "second");
		CHECK(first->read_string("../a/value.txt") == "first");
		CHECK(snapshot.size() == 1);
		CHECK(resources.accessed_resources() == std::vector<std::string>{resources.canonical_path("a/value.txt"), resources.canonical_path("b/value.txt")});
		{
			io::CheckpointWriter writer(directory / "second.h5", json::object(), metadata);
			write_test_checkpoint_state(writer);
			writer.embed_resources(*first);
			writer.finalize();
		}
		fs::remove_all(directory / "inputs");
		io::CheckpointReader initial(directory / "first.h5");
		CHECK(initial.resources().read_string("value.txt") == "first");
		CHECK_FALSE(initial.resources().exists("../b/value.txt"));
		io::CheckpointReader complete(directory / "second.h5");
		CHECK(complete.resources().read_string("value.txt") == "first");
		CHECK(complete.resources().with_root("../b")->read_string("value.txt") == "second");
		io::CheckpointWriter missing(directory / "missing.h5", json::object(), metadata);
		CHECK_THROWS(missing.embed_resources(*first));
	}
	fs::remove_all(directory);
}

TEST_CASE("Checkpoint embedding preserves typed HDF5 objects", "[hdf5][resource_io][checkpoint]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path directory = fs::temp_directory_path() / "polyfem-checkpoint-typed-resources";
	fs::create_directories(directory);
	const fs::path input = directory / "input.h5";
	const fs::path checkpoint = directory / "checkpoint.h5";
	Eigen::MatrixXd matrix(2, 3);
	matrix << -1.25, 0.5, 1024.75, 3.125, -8.5, 0.0625;
	const std::vector<int64_t> indices{int64_t(1) << 54, -300, 512};
	{
		h5pp::File file(input.string(), h5pp::FileAccess::REPLACE);
		file.writeDataset(matrix, "/data/weights/values");
		file.writeDataset(indices, "/data/weights/indices");
		file.writeAttribute(std::array<long, 2>{2, 3}, "/data/weights", "shape");
		file.writeAttribute(0.125, "/data/weights/values", "scale");
		file.writeAttribute(std::string("custom"), "/data/weights", "description");
		file.writeAttribute(long(0), "/data/weights", "elements_are_ordered");
		file.writeDataset(matrix, "/data/standalone");
		file.writeAttribute(std::string("matrix"), "/data/standalone", "description");
		file.writeDataset(std::string("first"), "/a/value.txt");
		file.writeDataset(std::string("second"), "/b/value.txt");
	}
	{
		io::HDF5IO resources(input);
		const auto data = resources.with_root("data");
		CHECK(data->read_matrix("weights/values") == matrix);
		CHECK(data->read_shape_attribute("weights", "shape") == std::array<long, 2>{2, 3});
		CHECK(data->read_matrix("standalone") == matrix);
		CHECK(resources.with_root("a")->read_string("value.txt") == "first");
		CHECK(resources.with_root("b")->read_string("value.txt") == "second");
		const auto manifest = resources.accessed_resources();
		CHECK(std::count(manifest.begin(), manifest.end(), "/a/value.txt") == 1);
		CHECK(std::count(manifest.begin(), manifest.end(), "/b/value.txt") == 1);
		io::CheckpointMetadata metadata;
		metadata.dt = 0.1;
		io::CheckpointWriter writer(checkpoint, json::object(), metadata);
		write_test_checkpoint_state(writer);
		writer.embed_resources(*data);
		writer.finalize();
	}
	fs::remove(input);
	{
		io::CheckpointReader reader(checkpoint);
		CHECK(reader.resources().read_matrix("weights/values") == matrix);
		CHECK(reader.resources().read_matrix("standalone") == matrix);
		CHECK(reader.resources().read_shape_attribute("weights", "shape") == std::array<long, 2>{2, 3});
		CHECK(reader.resources().read_string_attribute("weights", "description") == "custom");
		CHECK(reader.resources().read_integer_attribute("weights", "elements_are_ordered") == 0);
		CHECK(reader.resources().read_string("/a/value.txt") == "first");
		CHECK(reader.resources().read_string("/b/value.txt") == "second");
		h5pp::File file(checkpoint.string(), h5pp::FileAccess::READONLY);
		CHECK(file.readDataset<std::vector<int64_t>>("/resources/tree/data/weights/indices") == indices);
		CHECK(file.readAttribute<double>("/resources/tree/data/weights/values", "scale") == 0.125);
		CHECK(file.readAttribute<std::string>("/resources/tree/data/standalone", "description") == "matrix");
		const auto type = file.getTypeInfoDataset("/resources/tree/data/weights/indices");
		REQUIRE(type.h5Type.has_value());
		CHECK(H5Tget_class(type.h5Type.value()) == H5T_INTEGER);
		CHECK(H5Tget_size(type.h5Type.value()) == sizeof(int64_t));
	}
	fs::remove_all(directory);
}

TEST_CASE("Materialization keeps equal basenames distinct", "[hdf5][resource_io]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path input = fs::temp_directory_path() / "polyfem-materialize-basenames.h5";
	{
		h5pp::File file(input.string(), h5pp::FileAccess::REPLACE);
		file.writeDataset(std::string("first"), "/a/mesh.obj");
		file.writeDataset(std::string("second"), "/b/mesh.obj");
	}
	fs::path first_path, second_path;
	{
		io::HDF5IO resources(input);
		const auto first = resources.with_root("a");
		const auto second = resources.with_root("b");
		first_path = first->materialize("mesh.obj");
		second_path = second->materialize("mesh.obj");
		CHECK(first_path != second_path);
		CHECK(first_path.extension() == ".obj");
		CHECK(second_path.extension() == ".obj");
		CHECK(resources.materialize("/a/mesh.obj") == first_path);
		const io::FileSystemIO filesystem(fs::temp_directory_path());
		CHECK(filesystem.read_string(first_path.string()) == "first");
		CHECK(filesystem.read_string(second_path.string()) == "second");
	}
	CHECK_FALSE(fs::exists(first_path));
	CHECK_FALSE(fs::exists(second_path));
	fs::remove(input);
}

TEST_CASE("Checkpoint publication preserves an existing destination on failure", "[hdf5][checkpoint]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path directory = fs::temp_directory_path() / "polyfem-checkpoint-publication";
	fs::create_directories(directory);
	const fs::path target = directory / "checkpoint.h5";
	io::CheckpointMetadata metadata;
	{
		io::CheckpointWriter writer(target, json{{"generation", 1}}, metadata);
		writer.finalize();
	}
	{
		io::CheckpointWriter writer(target, json{{"generation", 2}}, metadata);
		writer.finalize();
		writer.finalize();
	}
	{
		io::HDF5IO published(target);
		CHECK(json::parse(published.read_string("/config"))["generation"] == 2);
	}
	{
		io::CheckpointWriter writer(target, json{{"generation", 3}}, metadata);
		// Simulate losing the unpublished temporary file before rename.
		fs::path temporary;
		for (const auto &entry : fs::directory_iterator(directory))
			if (entry.path().extension() == ".tmp")
				temporary = entry.path();
		REQUIRE_FALSE(temporary.empty());
		fs::rename(temporary, directory / "unpublished.h5");
		CHECK_THROWS(writer.finalize());
		REQUIRE(fs::is_regular_file(target));
		io::HDF5IO published(target);
		CHECK(json::parse(published.read_string("/config"))["generation"] == 2);
	}
	const fs::path occupied = directory / "occupied.h5";
	fs::create_directory(occupied);
	{
		io::CheckpointWriter writer(occupied, json::object(), metadata);
		CHECK_THROWS(writer.finalize());
		CHECK(fs::is_directory(occupied));
	}
	fs::remove_all(directory);
}

TEST_CASE("State applies a relative resource root once", "[resource_io][state]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path directory = fs::current_path() / "polyfem-relative-root-test";
	fs::create_directories(directory);
	std::ofstream(directory / "triangle.obj") << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
	json args = {
		{"root_path", "polyfem-relative-root-test"},
		{"geometry", {{"mesh", "triangle.obj"}}},
		{"materials", {{"type", "Laplacian"}}},
		{"output", {{"log", {{"quiet", true}}}}}};
	{
		State state;
		state.init(args, true);
		CHECK_NOTHROW(state.load_mesh());
		// Validation restores the empty default, not the consumed loader hint.
		CHECK(state.args.at("root_path") == "");
	}
	fs::remove_all(directory);
}

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
	{
		std::ofstream out(directory / "nested" / "input.yaml");
		out << "root_path: .\nvalue: 17\n";
	}
	{
		std::ofstream out(directory / "nested" / "common.json");
		out << R"({"common_value":23})";
	}
	io::FileSystemIO filesystem(directory);
	CHECK(filesystem.read_string("nested/resource.txt") == "resource contents");
	CHECK(filesystem.with_root("nested")->exists("resource.txt"));
	CHECK(filesystem.glob("nested/*.txt") == std::vector<std::string>{"nested/resource.txt"});
	json filesystem_config = {{"common", "nested/common.json"}, {"local_value", 29}};
	CHECK(utils::apply_common_params(filesystem_config, filesystem) == nullptr);
	CHECK(filesystem_config["common_value"] == 23);
	CHECK(filesystem_config["local_value"] == 29);
	const io::LoadedInput yaml = io::load_yaml_input(directory / "nested" / "input.yaml");
	CHECK(yaml.config == json{{"value", 17}});
	CHECK(yaml.resources->read_string("resource.txt") == "resource contents");

	const fs::path bundle = directory / "bundle.h5";
	{
		h5pp::File file(bundle.string(), h5pp::FileAccess::REPLACE);
		file.writeDataset(std::string(R"({"common":"configs/common.json","geometry":[]})"), "/config");
		file.writeDataset(std::string(R"({"root_path":".","materials":{"type":"NeoHookean"}})"), "/configs/common.json");
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
	CHECK(loaded.resources->with_root("assets")->read_string("note.txt") == "embedded text");
	CHECK(loaded.resources->glob("assets/*.txt") == std::vector<std::string>{"assets/note.txt"});
	{
		auto stream = loaded.resources->open("assets/note.txt", false);
		std::string contents;
		std::getline(*stream, contents);
		CHECK(contents == "embedded text");
	}
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
	const mesh::MeshData polyhedron_data = polyhedron->to_mesh_data();
	REQUIRE(polyhedron_data.has_polyhedral_topology());
	CHECK(polyhedron_data.faces.size() == 4);
	CHECK(polyhedron_data.cell_faces == std::vector<std::vector<int>>{{0, 1, 2, 3}});
	CHECK(polyhedron_data.cell_face_orientations == std::vector<std::vector<int>>{{1, 1, 1, 1}});
	CHECK(polyhedron_data.cell_kernel_points.isApprox(Eigen::RowVector3d(0.25, 0.25, 0.25)));
	const auto restored_polyhedron = mesh::Mesh::create(polyhedron_data);
	const auto *restored_conforming_polyhedron = dynamic_cast<const mesh::CMesh3D *>(restored_polyhedron.get());
	REQUIRE(restored_conforming_polyhedron != nullptr);
	CHECK(restored_conforming_polyhedron->kernel(0).isApprox(Eigen::RowVector3d(0.25, 0.25, 0.25)));

	const fs::path legacy_bundle = directory / "legacy-config-key.h5";
	{
		h5pp::File file(legacy_bundle.string(), h5pp::FileAccess::REPLACE);
		file.writeDataset(std::string(R"({"geometry":[]})"), "/json");
	}
	CHECK(io::load_hdf5_input(legacy_bundle).config == json{{"geometry", json::array()}});
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
	mesh::MeshData mesh_data(vertices, cells);
	mesh_data.body_ids = {7};
	mesh_data.geometry_ids = {9};
	mesh_data.node_ids = {21, 22, 23};
	mesh_data.boundary_elements = {{0, 1}, {1, 2}, {2, 0}};
	mesh_data.boundary_ids = {11, 12, 13};
	auto mesh = mesh::Mesh::create(mesh_data, false);
	REQUIRE(mesh != nullptr);
	const io::FileSystemIO resources(POLYFEM_DATA_DIR);
	auto higher_order_mesh = mesh::MeshLoader(resources).load_fem("contact/meshes/3D/simple/sphere/coarse/P2.msh");
	REQUIRE(higher_order_mesh != nullptr);
	const mesh::MeshData higher_order_data = higher_order_mesh->to_mesh_data();
	REQUIRE_FALSE(higher_order_data.higher_order_connectivity.empty());
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
		writer.write_mesh("/checkpoint/meshes/high_order", *higher_order_mesh);
		writer.write_matrix("/checkpoint/state/solution", Eigen::MatrixXd::Ones(3, 1));
		writer.finalize();
	}
	io::CheckpointReader reader(path);
	CHECK(reader.metadata().schema_version == io::CHECKPOINT_SCHEMA_VERSION);
	CHECK(reader.metadata().formulation == "Laplacian");
	CHECK(reader.metadata().step == 3);
	CHECK(reader.read_matrix("/checkpoint/state/solution").isOnes());
	const auto restored = reader.read_mesh("/checkpoint/meshes/active");
	REQUIRE(restored != nullptr);
	CHECK(restored->get_body_id(0) == 7);
	CHECK(restored->get_geometry_id(0) == 9);
	CHECK(restored->get_node_id(2) == 23);
	std::vector<int> restored_boundary_ids;
	for (int edge = 0; edge < restored->n_edges(); ++edge)
		restored_boundary_ids.push_back(restored->get_boundary_id(edge));
	std::sort(restored_boundary_ids.begin(), restored_boundary_ids.end());
	CHECK(restored_boundary_ids == std::vector<int>{11, 12, 13});

	const auto restored_higher_order = reader.read_mesh("/checkpoint/meshes/high_order");
	REQUIRE(restored_higher_order != nullptr);
	const mesh::MeshData restored_higher_order_data = restored_higher_order->to_mesh_data();
	CHECK(restored_higher_order_data.vertices.isApprox(higher_order_data.vertices));
	CHECK(restored_higher_order_data.elements == higher_order_data.elements);
	CHECK(restored_higher_order_data.higher_order_nodes.isApprox(higher_order_data.higher_order_nodes));
	CHECK(restored_higher_order_data.higher_order_connectivity == higher_order_data.higher_order_connectivity);
	fs::remove(path);
}

TEST_CASE("Checkpoint reader rejects corrupt schemas", "[hdf5][checkpoint]")
{
	namespace fs = std::filesystem;
	using namespace polyfem;
	const fs::path directory = fs::temp_directory_path() / "polyfem-invalid-checkpoints";
	fs::remove_all(directory);
	fs::create_directories(directory);

	const auto write_fixture = [&](
								   const fs::path &path,
								   const long checkpoint_version,
								   const long mesh_version,
								   const bool include_remaining_steps,
								   const bool include_cells) {
		h5pp::File file(path.string(), h5pp::FileAccess::REPLACE);
		file.writeDataset(std::string(R"({"time":{"dt":0.1}})"), "/config");
		file.writeDataset(std::string("/"), "/resources/root");
		file.writeDataset(checkpoint_version, "/checkpoint/metadata/schema_version");
		file.writeDataset(std::string("Laplacian"), "/checkpoint/metadata/formulation");
		file.writeDataset(long(1), "/checkpoint/metadata/step");
		file.writeDataset(0.1, "/checkpoint/metadata/time");
		file.writeDataset(0.1, "/checkpoint/metadata/dt");
		if (include_remaining_steps)
			file.writeDataset(long(1), "/checkpoint/metadata/remaining_steps");
		file.writeDataset(long(1), "/checkpoint/metadata/output_index");
		Eigen::MatrixXd vertices(3, 2);
		vertices << 0, 0, 1, 0, 0, 1;
		file.writeDataset(vertices, "/checkpoint/meshes/active/vertices");
		if (include_cells)
		{
			Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> cells(1, 3);
			cells << 0, 1, 2;
			file.writeDataset(cells, "/checkpoint/meshes/active/cells");
		}
		file.writeAttribute(mesh_version, "/checkpoint/meshes/active", "schema_version");
		file.writeAttribute(long(2), "/checkpoint/meshes/active", "dimension");
		file.writeAttribute(std::string("fem"), "/checkpoint/meshes/active", "mesh_type");
		file.createGroup("/checkpoint/state");
	};

	const fs::path missing_metadata = directory / "missing-metadata.h5";
	write_fixture(
		missing_metadata, io::CHECKPOINT_SCHEMA_VERSION, mesh::MESH_SCHEMA_VERSION,
		/*include_remaining_steps=*/false, /*include_cells=*/true);
	CHECK_THROWS(io::CheckpointReader{missing_metadata});

	const fs::path wrong_checkpoint_version = directory / "wrong-checkpoint-version.h5";
	write_fixture(
		wrong_checkpoint_version, io::CHECKPOINT_SCHEMA_VERSION + 1, mesh::MESH_SCHEMA_VERSION,
		/*include_remaining_steps=*/true, /*include_cells=*/true);
	CHECK_THROWS(io::CheckpointReader{wrong_checkpoint_version});

	const fs::path wrong_mesh_version = directory / "wrong-mesh-version.h5";
	write_fixture(
		wrong_mesh_version, io::CHECKPOINT_SCHEMA_VERSION, mesh::MESH_SCHEMA_VERSION + 1,
		/*include_remaining_steps=*/true, /*include_cells=*/true);
	{
		const io::CheckpointReader reader(wrong_mesh_version);
		CHECK_THROWS(reader.read_mesh("/checkpoint/meshes/active"));
	}

	const fs::path missing_cells = directory / "missing-cells.h5";
	write_fixture(
		missing_cells, io::CHECKPOINT_SCHEMA_VERSION, mesh::MESH_SCHEMA_VERSION,
		/*include_remaining_steps=*/true, /*include_cells=*/false);
	{
		const io::CheckpointReader reader(missing_cells);
		CHECK_THROWS(reader.read_mesh("/checkpoint/meshes/active"));
	}

	fs::remove_all(directory);
}
