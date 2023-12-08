#include "GeogramUtils.hpp"

#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

namespace polyfem::utils
{
	namespace
	{
		class GeoLoggerForward : public GEO::LoggerClient
		{
			std::shared_ptr<spdlog::logger> logger_;

		public:
			template <typename T>
			GeoLoggerForward(T logger) : logger_(logger) {}

		private:
			std::string truncate(const std::string &msg)
			{
				static size_t prefix_len = GEO::CmdLine::ui_feature(" ", false).size();
				return msg.substr(prefix_len, msg.size() - 1 - prefix_len);
			}

		protected:
			void div(const std::string &title) override
			{
				logger_->trace(title.substr(0, title.size() - 1));
			}

			void out(const std::string &str) override
			{
				logger_->info(truncate(str));
			}

			void warn(const std::string &str) override
			{
				logger_->warn(truncate(str));
			}

			void err(const std::string &str) override
			{
				logger_->error(truncate(str));
			}

			void status(const std::string &str) override
			{
				// Errors and warnings are also dispatched as status by geogram, but without
				// the "feature" header. We thus forward them as trace, to avoid duplicated
				// logger info...
				logger_->trace(str.substr(0, str.size() - 1));
			}
		};
	} // namespace

	void GeogramUtils::initialize()
	{
		if (initialized)
			return;

		initialized = true;

#ifndef WIN32
		setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif

		GEO::initialize();

		// Import standard command line arguments, and custom ones
		GEO::CmdLine::import_arg_group("standard");
		GEO::CmdLine::import_arg_group("pre");
		GEO::CmdLine::import_arg_group("algo");
	}

	void GeogramUtils::set_logger(spdlog::logger &logger)
	{
		GEO::Logger *geo_logger = GEO::Logger::instance();
		geo_logger->unregister_all_clients();
		geo_logger->register_client(new GeoLoggerForward(logger.clone("geogram")));
		geo_logger->set_pretty(false);
	}

} // namespace polyfem::utils