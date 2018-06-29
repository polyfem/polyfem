#ifndef COMMAND_LINE_HPP
#define COMMAND_LINE_HPP

#include <map>
#include <string>
#include <functional>

namespace polyfem {

	class CommandLine {
	public:
		typedef std::function<void(const int, const char**)> Callback;

		inline void add_callback(const std::string &key, Callback callback)
		{
			callback_[key] = callback;
		}


		inline void add_option(const std::string &key, double &value)
		{
			opt_args_double_[key] = &value;
		}

		inline void add_option(const std::string &key, float &value)
		{
			opt_args_float_[key] = &value;
		}

		inline void add_option(const std::string &key, int &value)
		{
			opt_args_int_[key] = &value;
		}

		inline void add_option(const std::string &yes_key,
							   const std::string &no_key,
							   bool &value)
		{
			opt_args_bool_[std::make_pair(yes_key, no_key)] = &value;
		}

		inline void add_option(const std::string &key, std::string &value)
		{
			opt_args_string_[key] = &value;
		}

		bool parse(const int argc, const char* argv[]);

		std::map<std::string, std::string *> opt_args_string_;
		std::map<std::pair<std::string, std::string>, bool *> opt_args_bool_;
		std::map<std::string, double *> opt_args_double_;
		std::map<std::string, float *> opt_args_float_;
		std::map<std::string, int *> opt_args_int_;

		std::map<std::string, Callback> callback_;
	};
}

#endif //COMMAND_LINE_HPP
