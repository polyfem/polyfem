#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <h5pp/h5pp.h>

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