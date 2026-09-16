#include "ResourceIO.hpp"
#include "InputLoader.hpp"

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/YamlToJson.hpp>
#include <polyfem/utils/Logger.hpp>

#include <h5pp/h5pp.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <type_traits>

namespace fs = std::filesystem;

namespace polyfem::io
{
	namespace
	{
		// Logical paths are POSIX paths, regardless of the host operating system.
		std::string normalize_logical(const std::string &path)
		{
			const bool absolute = !path.empty() && path.front() == '/';
			std::vector<std::string> components;
			std::istringstream input(path);
			std::string component;
			while (std::getline(input, component, '/'))
			{
				if (component.empty() || component == ".")
					continue;
				if (component == ".." && !components.empty() && components.back() != "..")
					components.pop_back();
				else if (component != ".." || !absolute)
					components.push_back(component);
			}
			std::string result = absolute ? "/" : "";
			for (const auto &part : components)
			{
				if (!result.empty() && result.back() != '/')
					result += '/';
				result += part;
			}
			return result.empty() ? "." : result;
		}

		std::string join_logical(const std::string &base, const std::string &path)
		{
			return normalize_logical(!path.empty() && path.front() == '/'
										 ? path
										 : (base.empty() ? "/" : base) + "/" + path);
		}

		bool is_windows_absolute(const std::string &path)
		{
			return path.size() >= 3
				   && ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z'))
				   && path[1] == ':' && (path[2] == '/' || path[2] == '\\');
		}

		std::string windows_separators(std::string path)
		{
			std::replace(path.begin(), path.end(), '\\', '/');
			return path;
		}

		std::string normalize_windows_absolute(const std::string &path)
		{
			const std::string normalized = windows_separators(path);
			return normalized.substr(0, 2) + normalize_logical(normalized.substr(2));
		}

		std::string relative_logical(const std::string &path, const std::string &base)
		{
			const auto split = [](const std::string &value) {
				std::vector<std::string> parts;
				std::istringstream input(value);
				std::string part;
				while (std::getline(input, part, '/'))
					if (!part.empty())
						parts.push_back(part);
				return parts;
			};
			const auto target = split(path), root = split(base);
			size_t common = 0;
			while (common < target.size() && common < root.size() && target[common] == root[common])
				++common;
			std::string result;
			for (size_t i = common; i < root.size(); ++i)
				result += "../";
			for (size_t i = common; i < target.size(); ++i)
				result += target[i] + "/";
			if (!result.empty())
				result.pop_back();
			return result.empty() ? "." : result;
		}

		template <typename T>
		std::vector<T> matrix_vector(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &m)
		{
			if (m.size() == 0)
				return {};
			return std::vector<T>(m.data(), m.data() + m.size());
		}

		template <typename T>
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> read_file_matrix(const fs::path &path)
		{
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result;
			if (!io::read_matrix(path.string(), result))
				log_and_throw_error("Unable to parse matrix resource {}", path.string());
			return result;
		}

		bool is_file_dataset(const h5pp::DsetInfo &info)
		{
			const hid_t type = info.h5Type.value();
			// The bundle format reserves rank-one uint8 datasets for file bytes.
			return H5Tget_class(type) == H5T_STRING
				   || (info.dsetRank.value() == 1 && H5Tget_class(type) == H5T_INTEGER
					   && H5Tget_size(type) == 1 && H5Tget_sign(type) == H5T_SGN_NONE);
		}

		template <typename T>
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> read_numeric_matrix(const h5pp::DsetInfo &info)
		{
			const std::string &path = info.dsetPath.value();
			const auto &dims = info.dsetDims.value();
			if (dims.size() > 2)
				log_and_throw_error("Numeric resource {} must be a scalar, vector or matrix.", path);
			const hsize_t rows = dims.empty() ? 1 : dims[0];
			const hsize_t cols = dims.size() < 2 ? 1 : dims[1];
			const hsize_t limit = std::numeric_limits<Eigen::Index>::max();
			if (rows > limit || cols > limit || (cols && rows > limit / cols))
				log_and_throw_error("Numeric resource {} is too large.", path);
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result(rows, cols);
			const hid_t type = info.h5Type.value();
			const H5T_class_t kind = H5Tget_class(type);
			const auto read = [&](auto scalar, const hid_t memory_type) {
				using Source = decltype(scalar);
				std::vector<Source> values(result.size());
				if (!values.empty() && H5Dread(info.h5Dset.value(), memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0)
					log_and_throw_error("Unable to read numeric resource {}.", path);
				for (Eigen::Index r = 0; r < result.rows(); ++r)
					for (Eigen::Index c = 0; c < result.cols(); ++c)
					{
						const Source value = values[r * result.cols() + c];
						if constexpr (std::is_integral_v<T> && std::is_integral_v<Source>)
						{
							if constexpr (std::is_signed_v<Source>)
							{
								if (value < std::numeric_limits<T>::lowest() || value > std::numeric_limits<T>::max())
									log_and_throw_error("Integer value in {} is out of range for the requested type.", path);
							}
							else if (value > uint64_t(std::numeric_limits<T>::max()))
								log_and_throw_error("Integer value in {} is out of range for the requested type.", path);
						}
						result(r, c) = static_cast<T>(value);
					}
			};
			if (kind == H5T_INTEGER && H5Tget_size(type) <= sizeof(uint64_t))
			{
				if (H5Tget_sign(type) == H5T_SGN_NONE)
					read(uint64_t(0), H5T_NATIVE_UINT64);
				else
					read(int64_t(0), H5T_NATIVE_INT64);
			}
			else if (kind == H5T_FLOAT && std::is_floating_point_v<T>)
				read(double(0), H5T_NATIVE_DOUBLE);
			else
				log_and_throw_error("Numeric resource {} has an incompatible datatype.", path);
			return result;
		}
	} // namespace

	class ResourceIO::AccessTracker
	{
	public:
		mutable std::mutex mutex;
		std::set<std::string> paths;
	};

	ResourceIO::ResourceIO() : access_tracker_(std::make_shared<AccessTracker>()) {}

	void ResourceIO::record_access(const std::string &path) const
	{
		const std::string canonical = canonical_path(path);
		std::lock_guard<std::mutex> lock(access_tracker_->mutex);
		access_tracker_->paths.insert(canonical);
	}

	std::vector<std::string> ResourceIO::accessed_resources() const
	{
		std::lock_guard<std::mutex> lock(access_tracker_->mutex);
		return {access_tracker_->paths.begin(), access_tracker_->paths.end()};
	}

	std::vector<std::string> ResourceIO::glob(const std::string &input_pattern) const
	{
		const std::string pattern = normalize_logical(input_pattern);
		if (pattern.find_first_of("*?") == std::string::npos)
			return exists(pattern) ? std::vector<std::string>{pattern} : std::vector<std::string>{};
		const bool recursive = pattern.find("**") != std::string::npos;
		const auto depth = std::count(pattern.begin(), pattern.end(), '/');
		std::string expression;
		expression.reserve(pattern.size() * 2);
		for (size_t i = 0; i < pattern.size(); ++i)
		{
			const char c = pattern[i];
			if (c == '*' && i + 1 < pattern.size() && pattern[i + 1] == '*')
			{
				++i;
				if (i + 1 < pattern.size() && pattern[i + 1] == '/')
				{
					expression += "(?:[^/]+/)*";
					++i;
				}
				else
					expression += ".*";
			}
			else if (c == '*')
				expression += "[^/]*";
			else if (c == '?')
				expression += "[^/]";
			else
			{
				if (std::string(".^$|()[]{}+\\").find(c) != std::string::npos)
					expression += '\\';
				expression += c;
			}
		}
		const std::regex matcher("^" + expression + "$");
		const size_t wildcard = pattern.find_first_of("*?");
		const std::string prefix = wildcard == std::string::npos ? pattern : pattern.substr(0, wildcard);
		const size_t slash = prefix.rfind('/');
		const std::string root = slash == std::string::npos ? "." : prefix.substr(0, slash);

		std::vector<std::string> result;
		std::vector<std::string> pending{root.empty() ? "/" : root};
		while (!pending.empty())
		{
			const std::string current = pending.back();
			pending.pop_back();
			for (const std::string &child : list(current))
			{
				if (std::regex_match(child, matcher))
					result.push_back(child);
				if ((recursive || std::count(child.begin(), child.end(), '/') < depth) && is_group(child))
					pending.push_back(child);
			}
		}
		std::sort(result.begin(), result.end());
		return result;
	}

	std::string ResourceIO::read_string(const std::string &path) const
	{
		auto in = open(path, true);
		return std::string(std::istreambuf_iterator<char>(*in), std::istreambuf_iterator<char>());
	}

	FileSystemIO::FileSystemIO(const fs::path &root, const fs::path &host_directory)
	{
		fs::path candidate = root.empty() ? fs::current_path() : root;
		std::error_code error;
		if (fs::exists(candidate, error) && !fs::is_directory(candidate, error))
			candidate = candidate.parent_path();
		root_ = fs::absolute(candidate).lexically_normal();
		host_directory_ = host_directory.empty() ? root_ : fs::absolute(host_directory).lexically_normal();
	}

	std::unique_ptr<const ResourceIO> FileSystemIO::with_root(const std::string &root) const
	{
		auto result = std::make_unique<FileSystemIO>(resolve(root), host_directory_);
		result->access_tracker_ = access_tracker_;
		return result;
	}

	fs::path FileSystemIO::resolve(const std::string &path) const
	{
		if (path.empty())
			return root_;
		fs::path p(path);
		return (p.is_absolute() ? p : root_ / p).lexically_normal();
	}

	bool FileSystemIO::exists(const std::string &path) const { return fs::exists(resolve(path)); }
	bool FileSystemIO::is_group(const std::string &path) const { return fs::is_directory(resolve(path)); }

	std::vector<std::string> FileSystemIO::list(const std::string &path) const
	{
		const fs::path directory = resolve(path);
		if (!fs::is_directory(directory))
			return {};
		std::vector<std::string> result;
		const bool absolute_input = fs::path(path).is_absolute();
		for (const fs::directory_entry &entry : fs::directory_iterator(directory))
			result.push_back(absolute_input ? entry.path().generic_string() : entry.path().lexically_relative(root_).generic_string());
		std::sort(result.begin(), result.end());
		return result;
	}

	std::unique_ptr<std::istream> FileSystemIO::open(const std::string &path, const bool binary) const
	{
		record_access(path);
		auto in = std::make_unique<std::ifstream>(resolve(path), std::ios::in | (binary ? std::ios::binary : std::ios::openmode(0)));
		if (!*in)
			log_and_throw_error("Unable to open input resource {}", describe(path));
		return in;
	}

	Eigen::MatrixXd FileSystemIO::read_matrix(const std::string &path) const
	{
		record_access(path);
		return read_file_matrix<double>(resolve(path));
	}

	Eigen::MatrixXi FileSystemIO::read_int_matrix(const std::string &path) const
	{
		record_access(path);
		return read_file_matrix<int>(resolve(path));
	}

	std::vector<double> FileSystemIO::read_double_vector(const std::string &path) const { return matrix_vector(read_matrix(path)); }
	std::vector<int> FileSystemIO::read_int_vector(const std::string &path) const { return matrix_vector(read_int_matrix(path)); }
	std::vector<long> FileSystemIO::read_long_vector(const std::string &path) const
	{
		record_access(path);
		return matrix_vector(read_file_matrix<long>(resolve(path)));
	}

	bool FileSystemIO::has_attribute(const std::string &, const std::string &) const { return false; }
	long FileSystemIO::read_integer_attribute(const std::string &, const std::string &) const
	{
		log_and_throw_error("Attributes require a structured HDF5 resource");
	}
	std::string FileSystemIO::read_string_attribute(const std::string &, const std::string &) const
	{
		log_and_throw_error("Attributes require a structured HDF5 resource");
	}
	std::array<long, 2> FileSystemIO::read_shape_attribute(const std::string &, const std::string &) const
	{
		log_and_throw_error("Attributes require a structured HDF5 resource");
	}

	fs::path FileSystemIO::materialize(const std::string &path) const
	{
		record_access(path);
		return resolve(path);
	}
	std::string FileSystemIO::describe(const std::string &path) const { return resolve(path).string(); }

	class HDF5IO::Impl
	{
	public:
		explicit Impl(const fs::path &path)
			: file(path.string(), h5pp::FileAccess::READONLY) {}

		~Impl()
		{
			std::error_code error;
			if (!temporary_directory.empty())
				fs::remove_all(temporary_directory, error);
		}

		h5pp::File file;
		mutable std::mutex mutex;
		mutable fs::path temporary_directory;
		mutable std::map<std::string, fs::path> materialized;
	};

	HDF5IO::HDF5IO(
		const fs::path &file,
		const std::string &root,
		const fs::path &host_directory,
		const std::string &storage_root)
		: impl_(std::make_shared<Impl>(file)),
		  file_path_(fs::absolute(file).lexically_normal()),
		  root_(!storage_root.empty() && is_windows_absolute(root)
					? normalize_windows_absolute(root)
					: join_logical("/", root)),
		  storage_root_(storage_root.empty() || storage_root == "/"
							? std::string()
							: join_logical("/", storage_root)),
		  host_directory_(host_directory.empty() ? file_path_.parent_path() : host_directory) {}

	HDF5IO::~HDF5IO() = default;

	std::unique_ptr<const ResourceIO> HDF5IO::with_root(const std::string &root) const
	{
		auto result = std::make_unique<HDF5IO>(
			file_path_, logical_resolve(root), host_directory_, storage_root_);
		result->impl_ = impl_;
		result->access_tracker_ = access_tracker_;
		return result;
	}

	std::string HDF5IO::logical_resolve(const std::string &path) const
	{
		// Mounted checkpoint resources retain the original filesystem namespace.
		// Ordinary HDF5 bundles still treat 'C:' and backslashes as POSIX names.
		if (!storage_root_.empty() && is_windows_absolute(root_))
		{
			const std::string normalized = windows_separators(path);
			if (is_windows_absolute(normalized))
				return normalize_windows_absolute(normalized);
			if (!normalized.empty() && normalized.front() == '/')
				return normalize_windows_absolute(root_.substr(0, 2) + normalized);
			return normalize_windows_absolute(root_ + "/" + normalized);
		}
		return join_logical(root_, path);
	}

	std::string HDF5IO::resolve(const std::string &path) const
	{
		const std::string logical = logical_resolve(path);
		return storage_root_.empty()
				   ? logical
				   : join_logical(storage_root_, logical.front() == '/' ? logical.substr(1) : logical);
	}
	bool HDF5IO::exists(const std::string &path) const { return impl_->file.linkExists(resolve(path)); }

	bool HDF5IO::is_group(const std::string &path) const
	{
		const std::string target = resolve(path);
		if (!impl_->file.linkExists(target))
			return false;
		const auto info = impl_->file.getLinkInfo(target);
		return info.h5ObjType.has_value() && info.h5ObjType.value() == H5O_TYPE_GROUP;
	}

	std::vector<std::string> HDF5IO::list(const std::string &path) const
	{
		if (!is_group(path))
			return {};
		const std::string parent = resolve(path);
		const std::string logical_parent = logical_resolve(path);
		const bool absolute_input = (!path.empty() && path.front() == '/')
									|| (!storage_root_.empty() && is_windows_absolute(root_) && is_windows_absolute(path));
		std::set<std::string> children;
		const auto add = [&](const std::vector<std::string> &entries) {
			for (const std::string &entry : entries)
			{
				std::string full = entry;
				if (full.empty())
					continue;
				if (full.front() != '/')
					full = join_logical(parent, full);
				const std::string relative = relative_logical(normalize_logical(full), parent);
				if (relative != "." && relative != ".." && relative.compare(0, 3, "../") != 0)
				{
					const std::string child = join_logical(logical_parent, relative.substr(0, relative.find('/')));
					children.insert(
						absolute_input ? child : relative_logical(child, root_));
				}
			}
		};
		add(impl_->file.findGroups("", parent));
		add(impl_->file.findDatasets("", parent));
		return {children.begin(), children.end()};
	}

	std::string HDF5IO::read_string(const std::string &path) const
	{
		record_access(path);
		const std::string key = resolve(path);
		const auto info = impl_->file.getDatasetInfo(key);
		if (!is_file_dataset(info))
			log_and_throw_error("Resource {} is typed numeric data, not a string or byte resource.", describe(path));
		if (H5Tget_class(info.h5Type.value()) == H5T_STRING)
			return impl_->file.readDataset<std::string>(key);
		const std::vector<unsigned char> bytes =
			impl_->file.readDataset<std::vector<unsigned char>>(key);
		return std::string(bytes.begin(), bytes.end());
	}

	std::unique_ptr<std::istream> HDF5IO::open(const std::string &path, const bool) const
	{
		return std::make_unique<std::istringstream>(read_string(path));
	}

	Eigen::MatrixXd HDF5IO::read_matrix(const std::string &path) const
	{
		record_access(path);
		const auto info = impl_->file.getDatasetInfo(resolve(path));
		return is_file_dataset(info) ? read_file_matrix<double>(materialize(path)) : read_numeric_matrix<double>(info);
	}

	Eigen::MatrixXi HDF5IO::read_int_matrix(const std::string &path) const
	{
		record_access(path);
		const auto info = impl_->file.getDatasetInfo(resolve(path));
		return is_file_dataset(info) ? read_file_matrix<int>(materialize(path)) : read_numeric_matrix<int>(info);
	}

	std::vector<double> HDF5IO::read_double_vector(const std::string &path) const
	{
		return matrix_vector(read_matrix(path));
	}

	std::vector<int> HDF5IO::read_int_vector(const std::string &path) const
	{
		return matrix_vector(read_int_matrix(path));
	}

	std::vector<long> HDF5IO::read_long_vector(const std::string &path) const
	{
		record_access(path);
		const auto info = impl_->file.getDatasetInfo(resolve(path));
		return matrix_vector(is_file_dataset(info) ? read_file_matrix<long>(materialize(path)) : read_numeric_matrix<long>(info));
	}

	bool HDF5IO::has_attribute(const std::string &path, const std::string &name) const
	{
		const std::string key = resolve(path);
		return impl_->file.attributeExists(std::string_view(key), std::string_view(name));
	}

	long HDF5IO::read_integer_attribute(const std::string &path, const std::string &name) const
	{
		record_access(path);
		return impl_->file.readAttribute<long>(resolve(path), name);
	}

	std::string HDF5IO::read_string_attribute(const std::string &path, const std::string &name) const
	{
		record_access(path);
		return impl_->file.readAttribute<std::string>(resolve(path), name);
	}

	std::array<long, 2> HDF5IO::read_shape_attribute(const std::string &path, const std::string &name) const
	{
		record_access(path);
		return impl_->file.readAttribute<std::array<long, 2>>(resolve(path), name);
	}

	fs::path HDF5IO::materialize(const std::string &path) const
	{
		record_access(path);
		const std::string key = resolve(path);
		std::lock_guard<std::mutex> lock(impl_->mutex);
		if (const auto it = impl_->materialized.find(key); it != impl_->materialized.end())
			return it->second;
		if (impl_->temporary_directory.empty())
		{
			const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
			for (size_t attempt = 0;; ++attempt)
			{
				const fs::path candidate = fs::temp_directory_path() / fmt::format("polyfem-resources-{}-{}", stamp, attempt);
				if (fs::create_directory(candidate))
				{
					impl_->temporary_directory = candidate;
					break;
				}
			}
		}
		// Separate directories preserve the original extension without aliasing
		// resources that happen to have the same basename.
		const fs::path directory = impl_->temporary_directory / std::to_string(impl_->materialized.size());
		fs::create_directory(directory);
		const fs::path filename = fs::path(key).filename();
		const fs::path output = directory / (filename.empty() ? fs::path("resource") : filename);
		std::ofstream file(output, std::ios::binary);
		const std::string contents = read_string(path);
		file.write(contents.data(), contents.size());
		if (!file)
			log_and_throw_error("Unable to materialize HDF5 resource {}", describe(path));
		impl_->materialized.emplace(key, output);
		return output;
	}

	std::string HDF5IO::describe(const std::string &path) const
	{
		return fmt::format("{}:{}", file_path_.string(), resolve(path));
	}

	namespace
	{
		LoadedInput apply_explicit_root(LoadedInput input)
		{
			if (input.config.contains("root_path") && input.config["root_path"].is_string()
				&& !input.config["root_path"].get<std::string>().empty())
			{
				input.resources = input.resources->with_root(input.config["root_path"]);
			}
			input.config.erase("root_path");
			return input;
		}
	} // namespace

	LoadedInput load_json_input(const fs::path &path)
	{
		auto resources = std::make_unique<FileSystemIO>(path.parent_path());
		LoadedInput result{json::parse(resources->read_string(path.filename().string())), std::move(resources)};
		return apply_explicit_root(std::move(result));
	}

	LoadedInput load_yaml_input(const fs::path &path)
	{
		auto resources = std::make_unique<FileSystemIO>(path.parent_path());
		LoadedInput result{yaml_string_to_json(resources->read_string(path.filename().string())), std::move(resources)};
		return apply_explicit_root(std::move(result));
	}

	LoadedInput load_hdf5_input(const fs::path &path)
	{
		auto resources = std::make_unique<HDF5IO>(path);
		std::string config_path = "/config";
		if (!resources->exists(config_path))
		{
			config_path = "/json";
			if (!resources->exists(config_path))
				log_and_throw_error("HDF5 input {} contains neither /config nor /json", path.string());
			logger().warn("HDF5 input {} uses deprecated /json; rename it to /config.", path.string());
		}
		LoadedInput result{json::parse(resources->read_string(config_path)), std::move(resources)};
		return apply_explicit_root(std::move(result));
	}
} // namespace polyfem::io
