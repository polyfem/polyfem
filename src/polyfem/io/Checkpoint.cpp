#include "Checkpoint.hpp"
#include "MeshWriter.hpp"

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/utils/Logger.hpp>

#include <h5pp/h5pp.h>

#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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
			// Canonical identities are already normalized. Keep Windows drive
			// names as components so dependencies on different drives cannot alias.
			const size_t start = logical.find_first_not_of('/');
			if (start == std::string::npos || logical == ".")
				return "/resources/tree";
			return "/resources/tree/" + logical.substr(start);
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
		write_string("/resources/root", "/");
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
	void CheckpointWriter::write_bytes(const std::string &path, const std::vector<unsigned char> &value) { impl_->file.writeDataset(value, path); }
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
		write_mesh(group, mesh.to_mesh_data());
	}

	void CheckpointWriter::write_mesh(const std::string &group, const mesh::MeshData &data)
	{
		io::write_mesh(impl_->file, group, data);
	}

	void CheckpointWriter::embed_resources(const ResourceIO &resources)
	{
		// Capture the manifest only once, after the completed step. Input reads
		// during solver setup (e.g. constraints) must be included as well.
		const std::vector<std::string> manifest = resources.accessed_resources();
		write_string("/resources/manifest", json(manifest).dump());
		impl_->file.deleteLink("/resources/root");
		write_string("/resources/root", resources.canonical_path(""));
		const auto *hdf5 = dynamic_cast<const HDF5IO *>(&resources);
		for (const std::string &canonical : manifest)
		{
			const std::string destination = resource_destination(canonical);
			// A previously copied group already contains its descendants.
			if (impl_->file.linkExists(destination))
				continue;
			if (!resources.exists(canonical))
				log_and_throw_error("Checkpoint dependency {} no longer exists.", resources.describe(canonical));
			if (hdf5)
			{
				// HDF5 object copying preserves datatype, dataspace, attributes and
				// group contents without converting numeric datasets to byte arrays.
				impl_->file.copyLinkFromFile(destination, hdf5->file_path(), hdf5->resolve(canonical));
				continue;
			}
			if (resources.is_group(canonical))
			{
				impl_->file.createGroup(destination);
				continue;
			}
			auto input = resources.open(canonical, true);
			const std::vector<unsigned char> contents{
				std::istreambuf_iterator<char>(*input), std::istreambuf_iterator<char>()};
			if (input->bad())
				log_and_throw_error("Unable to embed dependency {}.", resources.describe(canonical));
			write_bytes(destination, contents);
		}
	}

	void CheckpointWriter::finalize()
	{
		if (finalized_)
			return;
		impl_.reset();
		std::error_code error;
#ifdef _WIN32
		// Both paths are in the same directory, so this is a rename with
		// replacement, never a copy followed by deletion.
		if (!MoveFileExW(temporary_path_.c_str(), path_.c_str(), MOVEFILE_REPLACE_EXISTING))
			error = std::error_code(GetLastError(), std::system_category());
#else
		fs::rename(temporary_path_, path_, error);
#endif
		if (error)
			log_and_throw_error("Unable to atomically publish checkpoint {}: {}", path_.string(), error.message());
		finalized_ = true;
	}

	CheckpointReader::CheckpointReader(const fs::path &path)
		: path_(fs::absolute(path).lexically_normal()), io_(std::make_unique<HDF5IO>(path_))
	{
		const auto require = [&](const std::string &key) {
			if (!io_->exists(key))
				log_and_throw_error("Checkpoint {} is missing {}.", path_.string(), key);
		};
		for (const std::string &key : {
				 "/config", "/checkpoint/metadata/schema_version", "/checkpoint/metadata/formulation",
				 "/checkpoint/metadata/step", "/checkpoint/metadata/time", "/checkpoint/metadata/dt",
				 "/checkpoint/metadata/remaining_steps", "/checkpoint/metadata/output_index",
				 "/checkpoint/meshes/active", "/checkpoint/state", "/resources/root"})
			require(key);
		resources_ = std::make_unique<HDF5IO>(
			path_, io_->read_string("/resources/root"), path_.parent_path(), "/resources/tree");
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

	std::unique_ptr<mesh::Mesh> CheckpointReader::read_mesh(const std::string &path) const
	{
		return mesh::MeshLoader(*io_).load_fem(path);
	}
} // namespace polyfem::io
