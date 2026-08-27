#pragma once

#include <polyfem/Common.hpp>

#include <Eigen/Core>

#include <array>
#include <filesystem>
#include <istream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace polyfem::io
{
	/// Read-only interface to a hierarchical collection of simulation resources.
	class ResourceIO
	{
	public:
		virtual ~ResourceIO() = default;

		virtual std::unique_ptr<const ResourceIO> with_root(const std::string &root) const = 0;

		virtual bool exists(const std::string &path) const = 0;
		virtual bool is_group(const std::string &path) const = 0;
		virtual std::vector<std::string> list(const std::string &path) const = 0;
		/// Glob logical paths using POSIX '*' and '?' wildcards. '**' is recursive.
		std::vector<std::string> glob(const std::string &pattern) const;
		virtual std::unique_ptr<std::istream> open(const std::string &path, bool binary) const = 0;
		virtual std::string read_string(const std::string &path) const;
		virtual Eigen::MatrixXd read_matrix(const std::string &path) const = 0;
		virtual Eigen::MatrixXi read_int_matrix(const std::string &path) const = 0;
		virtual std::vector<double> read_double_vector(const std::string &path) const = 0;
		virtual std::vector<int> read_int_vector(const std::string &path) const = 0;
		virtual std::vector<long> read_long_vector(const std::string &path) const = 0;
		virtual bool has_attribute(const std::string &path, const std::string &name) const = 0;
		virtual long read_integer_attribute(const std::string &path, const std::string &name) const = 0;
		virtual std::string read_string_attribute(const std::string &path, const std::string &name) const = 0;
		virtual std::array<long, 2> read_shape_attribute(
			const std::string &path, const std::string &name) const = 0;
		virtual std::filesystem::path materialize(const std::string &path) const = 0;

		/// Host directory used only as the base for ordinary filesystem outputs.
		virtual const std::filesystem::path &host_directory() const = 0;
		virtual std::string describe(const std::string &path) const = 0;

		/// Resources actually consumed while preparing the simulation.
		std::vector<std::string> accessed_resources() const;
		void freeze_dependency_manifest() const;
		bool dependency_manifest_frozen() const;

	protected:
		class AccessTracker;
		ResourceIO();
		void record_access(const std::string &path) const;
		std::shared_ptr<AccessTracker> access_tracker_;
	};

	class FileSystemIO final : public ResourceIO
	{
	public:
		explicit FileSystemIO(
			const std::filesystem::path &root,
			const std::filesystem::path &host_directory = {});

		std::unique_ptr<const ResourceIO> with_root(const std::string &root) const override;
		bool exists(const std::string &path) const override;
		bool is_group(const std::string &path) const override;
		std::vector<std::string> list(const std::string &path) const override;
		std::unique_ptr<std::istream> open(const std::string &path, bool binary) const override;
		Eigen::MatrixXd read_matrix(const std::string &path) const override;
		Eigen::MatrixXi read_int_matrix(const std::string &path) const override;
		std::vector<double> read_double_vector(const std::string &path) const override;
		std::vector<int> read_int_vector(const std::string &path) const override;
		std::vector<long> read_long_vector(const std::string &path) const override;
		bool has_attribute(const std::string &path, const std::string &name) const override;
		long read_integer_attribute(const std::string &path, const std::string &name) const override;
		std::string read_string_attribute(const std::string &path, const std::string &name) const override;
		std::array<long, 2> read_shape_attribute(
			const std::string &path, const std::string &name) const override;
		std::filesystem::path materialize(const std::string &path) const override;
		const std::filesystem::path &host_directory() const override { return host_directory_; }
		std::string describe(const std::string &path) const override;

		std::filesystem::path resolve(const std::string &path) const;

	private:
		std::filesystem::path root_;
		std::filesystem::path host_directory_;
	};

	class HDF5IO final : public ResourceIO
	{
	public:
		explicit HDF5IO(
			const std::filesystem::path &file,
			const std::string &root = "/",
			const std::filesystem::path &host_directory = {});
		~HDF5IO() override;

		std::unique_ptr<const ResourceIO> with_root(const std::string &root) const override;
		bool exists(const std::string &path) const override;
		bool is_group(const std::string &path) const override;
		std::vector<std::string> list(const std::string &path) const override;
		std::unique_ptr<std::istream> open(const std::string &path, bool binary) const override;
		std::string read_string(const std::string &path) const override;
		Eigen::MatrixXd read_matrix(const std::string &path) const override;
		Eigen::MatrixXi read_int_matrix(const std::string &path) const override;
		std::vector<double> read_double_vector(const std::string &path) const override;
		std::vector<int> read_int_vector(const std::string &path) const override;
		std::vector<long> read_long_vector(const std::string &path) const override;
		bool has_attribute(const std::string &path, const std::string &name) const override;
		long read_integer_attribute(const std::string &path, const std::string &name) const override;
		std::string read_string_attribute(const std::string &path, const std::string &name) const override;
		std::array<long, 2> read_shape_attribute(
			const std::string &path, const std::string &name) const override;
		std::filesystem::path materialize(const std::string &path) const override;
		const std::filesystem::path &host_directory() const override { return host_directory_; }
		std::string describe(const std::string &path) const override;

		const std::filesystem::path &file_path() const { return file_path_; }
		std::string resolve(const std::string &path) const;

	private:
		class Impl;
		std::shared_ptr<Impl> impl_;
		std::filesystem::path file_path_;
		std::string root_;
		std::filesystem::path host_directory_;
	};

} // namespace polyfem::io
