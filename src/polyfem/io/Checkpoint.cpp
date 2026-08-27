#include "Checkpoint.hpp"

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/utils/Logger.hpp>

#include <h5pp/h5pp.h>

#include <chrono>
#include <fstream>

namespace fs = std::filesystem;

namespace polyfem::io
{
	class CheckpointWriter::Impl
	{
	public:
		explicit Impl(const fs::path &path) : file(path.string(), h5pp::FileAccess::REPLACE) {}
		h5pp::File file;
	};

	namespace
	{
		fs::path temporary_checkpoint_path(const fs::path &path)
		{
			const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
			return path.parent_path() / fmt::format(".{}.{}.tmp", path.filename().string(), stamp);
		}

		std::string resource_destination(const std::string &logical)
		{
			fs::path path(logical);
			if (path.is_absolute())
				path = path.relative_path();
			path = path.lexically_normal();
			if (path.empty() || path == ".")
				return "/resources/tree";
			for (const auto &part : path)
				if (part == "..")
					log_and_throw_error("Cannot embed resource outside its logical root: {}", logical);
			return (fs::path("/resources/tree") / path).generic_string();
		}

		std::string embedded_lookup_destination(const std::string &logical)
		{
			return fs::path(logical).is_absolute() ? fs::path(logical).lexically_normal().generic_string() : resource_destination(logical);
		}
	} // namespace

	CheckpointWriter::CheckpointWriter(const fs::path &path, const json &config, const CheckpointMetadata &metadata)
		: path_(fs::absolute(path).lexically_normal()), temporary_path_(temporary_checkpoint_path(path_))
	{
		if (path_.empty())
			log_and_throw_error("Checkpoint output path is empty.");
		fs::create_directories(path_.parent_path());
		impl_ = std::make_unique<Impl>(temporary_path_);
		write_string("/config", config.dump());
		write_long("/checkpoint/metadata/schema_version", metadata.schema_version);
		write_string("/checkpoint/metadata/formulation", metadata.formulation);
		write_long("/checkpoint/metadata/step", metadata.step);
		write_double("/checkpoint/metadata/time", metadata.time);
		write_double("/checkpoint/metadata/dt", metadata.dt);
		write_long("/checkpoint/metadata/remaining_steps", metadata.remaining_steps);
		write_long("/checkpoint/metadata/output_index", metadata.output_index);
	}

	CheckpointWriter::~CheckpointWriter()
	{
		impl_.reset();
		if (!finalized_)
		{
			std::error_code error;
			fs::remove(temporary_path_, error);
		}
	}

	void CheckpointWriter::write_matrix(const std::string &path, const Eigen::MatrixXd &value) { impl_->file.writeDataset(value, path); }
	void CheckpointWriter::write_int_matrix(const std::string &path, const Eigen::MatrixXi &value) { impl_->file.writeDataset(value.cast<int64_t>(), path); }
	void CheckpointWriter::write_vector(const std::string &path, const std::vector<double> &value) { impl_->file.writeDataset(value, path); }
	void CheckpointWriter::write_int_vector(const std::string &path, const std::vector<int> &value) { impl_->file.writeDataset(value, path); }
	void CheckpointWriter::write_long_vector(const std::string &path, const std::vector<long> &value) { impl_->file.writeDataset(value, path); }
	void CheckpointWriter::write_string(const std::string &path, const std::string &value) { impl_->file.writeDataset(value, path); }
	void CheckpointWriter::write_long(const std::string &path, const long value) { impl_->file.writeDataset(value, path); }
	void CheckpointWriter::write_double(const std::string &path, const double value) { impl_->file.writeDataset(value, path); }
	void CheckpointWriter::write_attribute(const std::string &path, const std::string &name, const long value) { impl_->file.writeAttribute(value, path, name); }
	void CheckpointWriter::write_attribute(const std::string &path, const std::string &name, const std::string &value) { impl_->file.writeAttribute(value, path, name); }

	void CheckpointWriter::write_mesh(const std::string &group, const mesh::Mesh &mesh)
	{
		const int dimension = mesh.dimension();
		Eigen::MatrixXd vertices(mesh.n_vertices(), dimension);
		for (int i = 0; i < mesh.n_vertices(); ++i)
			vertices.row(i) = mesh.point(i);
		if (mesh.n_elements() == 0)
			log_and_throw_error("Cannot checkpoint an empty mesh.");
		const int width = mesh.n_cell_vertices(0);
		Eigen::MatrixXi cells(mesh.n_elements(), width);
		for (int e = 0; e < mesh.n_elements(); ++e)
		{
			if (mesh.n_cell_vertices(e) != width)
				log_and_throw_error("Checkpoint typed mesh codec does not support mixed-width elements yet.");
			for (int j = 0; j < width; ++j)
				cells(e, j) = mesh.element_vertex(e, j);
		}
		write_matrix(group + "/vertices", vertices);
		write_int_matrix(group + "/cells", cells);
		write_attribute(group, "schema_version", mesh::MESH_SCHEMA_VERSION);
		write_attribute(group, "dimension", dimension);
		write_attribute(group, "mesh_type", "fem");
		if (mesh.has_body_ids())
			write_int_vector(group + "/body_ids", mesh.get_body_ids());
		if (mesh.has_geometry_ids())
			write_int_vector(group + "/geometry_ids", mesh.get_geometry_ids());
		if (mesh.has_boundary_ids())
		{
			std::vector<int> ids(mesh.n_boundary_elements());
			const int boundary_width = dimension == 3 ? mesh.n_face_vertices(0) : 2;
			Eigen::MatrixXi elements(mesh.n_boundary_elements(), boundary_width);
			for (int i = 0; i < mesh.n_boundary_elements(); ++i)
			{
				ids[i] = mesh.get_boundary_id(i);
				if (dimension == 3 && mesh.n_face_vertices(i) != boundary_width)
					log_and_throw_error("Checkpoint typed mesh codec does not support mixed-width boundary elements yet.");
				for (int j = 0; j < boundary_width; ++j)
					elements(i, j) = mesh.boundary_element_vertex(i, j);
			}
			write_int_matrix(group + "/boundary_elements", elements);
			write_int_vector(group + "/boundary_ids", ids);
		}
	}

	void CheckpointWriter::embed_resources(const ResourceIO &resources)
	{
		const std::vector<std::string> manifest = resources.accessed_resources();
		write_string("/resources/manifest", json(manifest).dump());
		for (const std::string &logical : manifest)
		{
			if (logical == "/config" || logical == "/json")
				continue;
			if (!resources.exists(logical))
				continue;
			const std::string destination = embedded_lookup_destination(logical);
			if (resources.is_group(logical))
			{
				impl_->file.createGroup(destination);
				for (const std::string &attribute : {"schema_version", "dimension"})
					if (resources.has_attribute(logical, attribute))
						write_attribute(destination, attribute, resources.read_integer_attribute(logical, attribute));
				if (resources.has_attribute(logical, "mesh_type"))
					write_attribute(destination, "mesh_type", resources.read_string_attribute(logical, "mesh_type"));
				continue;
			}
			if (dynamic_cast<const FileSystemIO *>(&resources) != nullptr)
			{
				const fs::path physical = resources.materialize(logical);
				std::ifstream input(physical, std::ios::binary);
				if (!input)
					log_and_throw_error("Unable to embed dependency {}.", resources.describe(logical));
				write_string(destination, std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()));
				continue;
			}
			try
			{
				write_string(destination, resources.read_string(logical));
			}
			catch (const std::exception &)
			{
				try
				{
					write_matrix(destination, resources.read_matrix(logical));
				}
				catch (const std::exception &)
				{
					write_int_matrix(destination, resources.read_int_matrix(logical));
				}
			}
		}
	}

	void CheckpointWriter::finalize()
	{
		if (finalized_)
			return;
		impl_.reset();
		std::error_code error;
		fs::rename(temporary_path_, path_, error);
		if (error)
		{
			fs::remove(path_, error);
			error.clear();
			fs::rename(temporary_path_, path_, error);
		}
		if (error)
			log_and_throw_error("Unable to atomically publish checkpoint {}: {}", path_.string(), error.message());
		finalized_ = true;
	}

	CheckpointReader::CheckpointReader(const fs::path &path)
		: path_(fs::absolute(path).lexically_normal()), io_(std::make_unique<HDF5IO>(path_)),
		  resources_(io_->with_root("/resources/tree"))
	{
		const auto require = [&](const std::string &key) {
			if (!io_->exists(key))
				log_and_throw_error("Checkpoint {} is missing {}.", path_.string(), key);
		};
		for (const std::string &key : {
				 "/config", "/checkpoint/metadata/schema_version", "/checkpoint/metadata/formulation",
				 "/checkpoint/metadata/step", "/checkpoint/metadata/time", "/checkpoint/metadata/dt",
				 "/checkpoint/metadata/remaining_steps", "/checkpoint/metadata/output_index",
				 "/checkpoint/meshes/active", "/checkpoint/state"})
			require(key);
		config_ = json::parse(io_->read_string("/config"));
		metadata_.schema_version = read_long("/checkpoint/metadata/schema_version");
		metadata_.formulation = read_string("/checkpoint/metadata/formulation");
		metadata_.step = read_long("/checkpoint/metadata/step");
		metadata_.time = read_double("/checkpoint/metadata/time");
		metadata_.dt = read_double("/checkpoint/metadata/dt");
		metadata_.remaining_steps = read_long("/checkpoint/metadata/remaining_steps");
		metadata_.output_index = read_long("/checkpoint/metadata/output_index");
		if (metadata_.schema_version != CHECKPOINT_SCHEMA_VERSION)
			log_and_throw_error(
				"Unsupported checkpoint schema {} in {}; expected {}.",
				metadata_.schema_version, path_.string(), CHECKPOINT_SCHEMA_VERSION);
		if (!(metadata_.dt > 0) || metadata_.step < 0 || metadata_.remaining_steps < 0)
			log_and_throw_error("Checkpoint {} has invalid temporal metadata.", path_.string());
	}

	Eigen::MatrixXd CheckpointReader::read_matrix(const std::string &path) const { return io_->read_matrix(path); }
	Eigen::MatrixXi CheckpointReader::read_int_matrix(const std::string &path) const { return io_->read_int_matrix(path); }
	std::vector<double> CheckpointReader::read_vector(const std::string &path) const { return io_->read_double_vector(path); }
	std::vector<int> CheckpointReader::read_int_vector(const std::string &path) const { return io_->read_int_vector(path); }
	std::vector<long> CheckpointReader::read_long_vector(const std::string &path) const { return io_->read_long_vector(path); }
	std::string CheckpointReader::read_string(const std::string &path) const { return io_->read_string(path); }
	long CheckpointReader::read_long(const std::string &path) const { return io_->read_long_vector(path).at(0); }
	double CheckpointReader::read_double(const std::string &path) const { return io_->read_double_vector(path).at(0); }

	std::unique_ptr<mesh::Mesh> CheckpointReader::read_mesh(const std::string &path, const bool non_conforming) const
	{
		return mesh::MeshLoader(*io_).load_fem(path, non_conforming);
	}
} // namespace polyfem::io
