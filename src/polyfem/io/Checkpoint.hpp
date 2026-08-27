#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/io/ResourceIO.hpp>

#include <Eigen/Core>

#include <filesystem>
#include <memory>
#include <string>

namespace polyfem::mesh
{
	class Mesh;
}

namespace polyfem::io
{
	inline constexpr long CHECKPOINT_SCHEMA_VERSION = 1;

	class CheckpointMetadata
	{
	public:
		long schema_version = CHECKPOINT_SCHEMA_VERSION;
		std::string formulation;
		long step = 0;
		double time = 0;
		double dt = 0;
		long remaining_steps = 0;
		long output_index = 0;
	};

	class CheckpointWriter
	{
	public:
		CheckpointWriter(const std::filesystem::path &path, const json &config, const CheckpointMetadata &metadata);
		~CheckpointWriter();

		CheckpointWriter(const CheckpointWriter &) = delete;
		CheckpointWriter &operator=(const CheckpointWriter &) = delete;

		void write_matrix(const std::string &path, const Eigen::MatrixXd &value);
		void write_int_matrix(const std::string &path, const Eigen::MatrixXi &value);
		void write_vector(const std::string &path, const std::vector<double> &value);
		void write_int_vector(const std::string &path, const std::vector<int> &value);
		void write_long_vector(const std::string &path, const std::vector<long> &value);
		void write_string(const std::string &path, const std::string &value);
		void write_long(const std::string &path, long value);
		void write_double(const std::string &path, double value);
		void write_attribute(const std::string &path, const std::string &name, long value);
		void write_attribute(const std::string &path, const std::string &name, const std::string &value);

		void write_mesh(const std::string &group, const mesh::Mesh &mesh);
		void embed_resources(const ResourceIO &resources);
		void finalize();

	private:
		class Impl;
		std::unique_ptr<Impl> impl_;
		std::filesystem::path path_;
		std::filesystem::path temporary_path_;
		bool finalized_ = false;
	};

	class CheckpointReader
	{
	public:
		explicit CheckpointReader(const std::filesystem::path &path);

		const std::filesystem::path &path() const { return path_; }
		const CheckpointMetadata &metadata() const { return metadata_; }
		const json &config() const { return config_; }
		const ResourceIO &resources() const { return *resources_; }
		std::unique_ptr<mesh::Mesh> read_mesh(const std::string &path, bool non_conforming = false) const;

		Eigen::MatrixXd read_matrix(const std::string &path) const;
		Eigen::MatrixXi read_int_matrix(const std::string &path) const;
		std::vector<double> read_vector(const std::string &path) const;
		std::vector<int> read_int_vector(const std::string &path) const;
		std::vector<long> read_long_vector(const std::string &path) const;
		std::string read_string(const std::string &path) const;
		long read_long(const std::string &path) const;
		double read_double(const std::string &path) const;
		bool exists(const std::string &path) const { return io_->exists(path); }

	private:
		std::filesystem::path path_;
		std::unique_ptr<const HDF5IO> io_;
		std::unique_ptr<const ResourceIO> resources_;
		json config_;
		CheckpointMetadata metadata_;
	};
} // namespace polyfem::io
