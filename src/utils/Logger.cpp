#include <polyfem/Logger.hpp>
#include <polyfem/DisableWarnings.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/details/registry.h>
#include <spdlog/details/thread_pool.h>
#include <polyfem/EnableWarnings.hpp>
#include <memory>
#include <mutex>
#include <iostream>

namespace polyfem {
	std::shared_ptr<spdlog::async_logger> Logger::logger_;


	namespace
	{
		// Some code was copied over from <spdlog/async.h>
		void aux_init(std::vector<spdlog::sink_ptr> &sinks) {
			auto &registry_inst = spdlog::details::registry::instance();

			// create global thread pool if not already exists..
			std::lock_guard<std::recursive_mutex> tp_lock(registry_inst.tp_mutex());
			auto tp = registry_inst.get_tp();
			if (tp == nullptr) {
				tp = std::make_shared<spdlog::details::thread_pool>(spdlog::details::default_async_q_size, 1);
				registry_inst.set_tp(tp);
			}

			Logger::logger_ = std::make_shared<spdlog::async_logger>("polyfem", sinks.begin(), sinks.end(), std::move(tp), spdlog::async_overflow_policy::block);
			registry_inst.register_and_init(Logger::logger_);
		}
	}

	void Logger::init(bool use_cout, const std::string &filename, bool truncate) {
		std::vector<spdlog::sink_ptr> sinks;
		if (use_cout) {
			sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
		}
		if (!filename.empty()) {
			sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, truncate));
		}

		aux_init(sinks);
	}


	void Logger::init(std::ostream &os) {
		std::vector<spdlog::sink_ptr> sinks;
		sinks.emplace_back(std::make_shared<spdlog::sinks::ostream_sink_mt>(os, true));

		aux_init(sinks);
	}



}
