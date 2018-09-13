#include <polyfem/CommandLine.hpp>
#include <polyfem/Logger.hpp>

#include <iostream>
#include <sstream>

namespace polyfem {

	template<typename Key>
	static bool aux_parse(const int argc, const char* argv[], std::map<std::string, Key *> &args)
	{
		for(auto &arg : args) {
			for(int i = 1; i < argc; ++i) {
				if(argv[i] == arg.first) {
					if(argc <= i + 1) {
						logger().error("malformed input for arg {}", argv[i]);
						return false;
					}

					std::istringstream ss(argv[i+1]);
					ss >> *arg.second;
					++i;
				}
			}
		}

		return true;
	}


	static bool aux_parse_boolean(const int argc, const char* argv[],
		std::map<std::pair<std::string, std::string>, bool *> &args)
	{
		for(auto &arg : args) {
			for(int i = 1; i < argc; ++i) {
				if(argv[i] == arg.first.first) {
					*arg.second = true;
				} else if(argv[i] == arg.first.second) {
					*arg.second = false;
				}
			}
		}

		return true;
	}


	static bool aux_parse_callback(const int argc, const char* argv[],
								    const std::map<std::string, CommandLine::Callback> &callback)
	{
		// for(auto &c : callback) {
			for(int i = 1; i < argc; ++i) {
				// if(argv[i] == c.first) {
					// (c.second)(argc, argv);
				// }
				auto it = callback.find(argv[i]);
				if(it == callback.end()) continue;
				it->second(argc, argv);
			}
		// }

		return true;
	}


	bool CommandLine::parse(const int argc, const char* argv[])
	{
		return aux_parse(argc, argv, opt_args_string_) &&
		aux_parse(argc, argv, opt_args_double_) &&
		aux_parse(argc, argv, opt_args_float_)  &&
		aux_parse(argc, argv, opt_args_int_)    &&
		aux_parse_boolean(argc, argv, opt_args_bool_) &&
		aux_parse_callback(argc, argv, callback_);
	}
}
