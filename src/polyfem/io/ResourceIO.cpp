#include "ResourceIO.hpp"
#include "InputLoader.hpp"

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/YamlToJson.hpp>
#include <polyfem/utils/Logger.hpp>

#include <h5pp/h5pp.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

namespace polyfem::io
{
	namespace
	{
		std::string join_logical(const std::string &base, const std::string &path)
		{
			if (path.empty())
				return base.empty() ? "/" : base;
			fs::path p(path);
			if (p.is_absolute())
				return p.lexically_normal().generic_string();
			return (fs::path(base.empty() ? "/" : base) / p).lexically_normal().generic_string();
		}

		template <typename T>
		std::vector<T> matrix_vector(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &m)
		{
			return std::vector<T>(m.data(), m.data() + m.size());
		}
	} // namespace

	class ResourceIO::AccessTracker
	{
	public:
		mutable std::mutex mutex;
		std::set<std::string> paths;
		bool frozen = false;
	};

	ResourceIO::ResourceIO() : access_tracker_(std::make_shared<AccessTracker>()) {}

	void ResourceIO::record_access(const std::string &path) const
	{
		std::lock_guard<std::mutex> lock(access_tracker_->mutex);
		if (!access_tracker_->frozen)
			access_tracker_->paths.insert(path);
	}

	std::vector<std::string> ResourceIO::accessed_resources() const
	{
		std::lock_guard<std::mutex> lock(access_tracker_->mutex);
		return {access_tracker_->paths.begin(), access_tracker_->paths.end()};
	}

	void ResourceIO::freeze_dependency_manifest() const
	{
		std::lock_guard<std::mutex> lock(access_tracker_->mutex);
		access_tracker_->frozen = true;
	}

	bool ResourceIO::dependency_manifest_frozen() const
	{
		std::lock_guard<std::mutex> lock(access_tracker_->mutex);
		return access_tracker_->frozen;
	}

	std::vector<std::string> ResourceIO::glob(const std::string &pattern) const
	{
		std::string expression;
		expression.reserve(pattern.size() * 2);
		for (size_t i = 0; i < pattern.size(); ++i)
		{
			const char c = pattern[i];
			if (c == '*' && i + 1 < pattern.size() && pattern[i + 1] == '*')
			{
				expression += ".*";
				++i;
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
				if (is_group(child))
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
		if (fs::exists(candidate) && !fs::is_directory(candidate))
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
		Eigen::MatrixXd result;
		if (!io::read_matrix(resolve(path).string(), result))
			log_and_throw_error("Unable to read matrix resource {}", describe(path));
		return result;
	}

	Eigen::MatrixXi FileSystemIO::read_int_matrix(const std::string &path) const
	{
		record_access(path);
		Eigen::MatrixXi result;
		if (!io::read_matrix(resolve(path).string(), result))
			log_and_throw_error("Unable to read integer matrix resource {}", describe(path));
		return result;
	}

	std::vector<double> FileSystemIO::read_double_vector(const std::string &path) const { return matrix_vector(read_matrix(path)); }
	std::vector<int> FileSystemIO::read_int_vector(const std::string &path) const { return matrix_vector(read_int_matrix(path)); }
	std::vector<long> FileSystemIO::read_long_vector(const std::string &path) const
	{
		const Eigen::MatrixXi m = read_int_matrix(path);
		return std::vector<long>(m.data(), m.data() + m.size());
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

	HDF5IO::HDF5IO(const fs::path &file, const std::string &root, const fs::path &host_directory)
		: impl_(std::make_shared<Impl>(file)),
		  file_path_(fs::absolute(file).lexically_normal()),
		  root_(join_logical("/", root)),
		  host_directory_(host_directory.empty() ? file_path_.parent_path() : host_directory) {}

	HDF5IO::~HDF5IO() = default;

	std::unique_ptr<const ResourceIO> HDF5IO::with_root(const std::string &root) const
	{
		auto result = std::make_unique<HDF5IO>(file_path_, resolve(root), host_directory_);
		result->impl_ = impl_;
		result->access_tracker_ = access_tracker_;
		return result;
	}

	std::string HDF5IO::resolve(const std::string &path) const { return join_logical(root_, path); }
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
		const std::string parent = resolve(path);
		std::set<std::string> children;
		const auto add = [&](const std::vector<std::string> &entries) {
			for (const std::string &entry : entries)
			{
				std::string full = entry;
				if (full.empty())
					continue;
				if (full.front() != '/')
					full = join_logical(parent, full);
				const fs::path relative = fs::path(full).lexically_relative(parent);
				if (!relative.empty())
					children.insert(join_logical(parent, (*relative.begin()).string()));
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
		try
		{
			return impl_->file.readDataset<std::string>(key);
		}
		catch (const std::exception &)
		{
			const std::vector<unsigned char> bytes = impl_->file.readDataset<std::vector<unsigned char>>(key);
			return std::string(bytes.begin(), bytes.end());
		}
	}

	std::unique_ptr<std::istream> HDF5IO::open(const std::string &path, const bool) const
	{
		return std::make_unique<std::istringstream>(read_string(path));
	}

	Eigen::MatrixXd HDF5IO::read_matrix(const std::string &path) const
	{
		record_access(path);
		return impl_->file.readDataset<Eigen::MatrixXd>(resolve(path));
	}

	Eigen::MatrixXi HDF5IO::read_int_matrix(const std::string &path) const
	{
		record_access(path);
		using MatrixXl = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;
		return impl_->file.readDataset<MatrixXl>(resolve(path)).cast<int>();
	}

	std::vector<double> HDF5IO::read_double_vector(const std::string &path) const
	{
		record_access(path);
		try
		{
			return impl_->file.readDataset<std::vector<double>>(resolve(path));
		}
		catch (const std::exception &)
		{
			return matrix_vector(read_matrix(path));
		}
	}

	std::vector<int> HDF5IO::read_int_vector(const std::string &path) const
	{
		record_access(path);
		try
		{
			return impl_->file.readDataset<std::vector<int>>(resolve(path));
		}
		catch (const std::exception &)
		{
			return matrix_vector(read_int_matrix(path));
		}
	}

	std::vector<long> HDF5IO::read_long_vector(const std::string &path) const
	{
		record_access(path);
		try
		{
			return impl_->file.readDataset<std::vector<long>>(resolve(path));
		}
		catch (const std::exception &)
		{
			const std::vector<int> values = read_int_vector(path);
			return {values.begin(), values.end()};
		}
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
			impl_->temporary_directory = fs::temp_directory_path() / fmt::format("polyfem-resources-{}", stamp);
			fs::create_directories(impl_->temporary_directory);
		}
		fs::path output = impl_->temporary_directory / fs::path(key).filename();
		if (output.filename().empty())
			output /= "resource";
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
		LoadedInput result{yaml_file_to_json(path.string()), std::move(resources)};
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
