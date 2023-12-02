#pragma once

#include "Logger.hpp"

namespace polyfem::utils
{
	class GeogramUtils
	{

	public:
		static GeogramUtils &instance()
		{
			static GeogramUtils singleton;

			return singleton;
		}

		void initialize();

		void set_logger(spdlog::logger &logger);

	private:
		GeogramUtils() {}

		bool initialized = false;
	};
} // namespace polyfem::utils