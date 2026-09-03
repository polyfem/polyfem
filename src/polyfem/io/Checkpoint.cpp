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

		template <typename T>
		std::pair<std::vector<T>, std::vector<long>> pack_ragged(const std::vector<std::vector<T>> &rows)
		{
			std::pair<std::vector<T>, std::vector<long>> packed;
			packed.second.reserve(rows.size() + 1);
			packed.second.push_back(0);
			for (const auto &row : rows)
			{
				packed.first.insert(packed.first.end(), row.begin(), row.end());
				packed.second.push_back(packed.first.size());
			}
			return packed;
		}

		Eigen::MatrixXi pack_padded(const std::vector<std::vector<int>> &rows)
		{
			int width = 0;
			for (const auto &row : rows)
				width = std::max(width, int(row.size()));
			Eigen::MatrixXi packed = Eigen::MatrixXi::Constant(rows.size(), width, -1);
			for (int i = 0; i < rows.size(); ++i)
				for (int j = 0; j < rows[i].size(); ++j)
					packed(i, j) = rows[i][j];
			return packed;
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
		write_mesh(group, mesh.to_mesh_data());
	}

	void CheckpointWriter::write_mesh(const std::string &group, const mesh::MeshData &data)
	{
		data.validate();
		write_matrix(group + "/vertices", data.vertices);
		write_int_matrix(group + "/cells", data.elements);
		write_attribute(group, "schema_version", mesh::MESH_SCHEMA_VERSION);
		write_attribute(group, "dimension", data.dimension());
		write_attribute(group, "mesh_type", "fem");
		if (!data.body_ids.empty())
			write_int_vector(group + "/body_ids", data.body_ids);
		if (!data.geometry_ids.empty())
			write_int_vector(group + "/geometry_ids", data.geometry_ids);
		if (!data.node_ids.empty())
			write_int_vector(group + "/node_ids", data.node_ids);
		if (!data.boundary_ids.empty())
		{
			write_int_matrix(group + "/boundary_elements", pack_padded(data.boundary_elements));
			write_int_vector(group + "/boundary_ids", data.boundary_ids);
		}
		if (!data.higher_order_connectivity.empty())
		{
			const auto packed = pack_ragged(data.higher_order_connectivity);
			write_matrix(group + "/higher_order_nodes", data.higher_order_nodes);
			write_int_vector(group + "/higher_order_connectivity", packed.first);
			write_long_vector(group + "/higher_order_offsets", packed.second);
		}
		if (!data.higher_order_weights.empty())
		{
			const auto packed = pack_ragged(data.higher_order_weights);
			write_vector(group + "/higher_order_weights", packed.first);
			write_long_vector(group + "/higher_order_weight_offsets", packed.second);
		}
		if (data.has_polyhedral_topology())
		{
			const auto faces = pack_ragged(data.faces);
			const auto cell_faces = pack_ragged(data.cell_faces);
			const auto orientations = pack_ragged(data.cell_face_orientations);
			write_int_vector(group + "/faces", faces.first);
			write_long_vector(group + "/face_offsets", faces.second);
			write_int_vector(group + "/cell_faces", cell_faces.first);
			write_long_vector(group + "/cell_face_offsets", cell_faces.second);
			write_int_vector(group + "/cell_face_orientations", orientations.first);
			std::vector<int> is_hex(data.cell_is_hex.size());
			std::transform(data.cell_is_hex.begin(), data.cell_is_hex.end(), is_hex.begin(), [](const bool value) { return int(value); });
			write_int_vector(group + "/cell_is_hex", is_hex);
			write_matrix(group + "/cell_kernel_points", data.cell_kernel_points);
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
