#pragma once

#include <spdlog/fmt/ostr.h>

#include <string>

namespace polyfem
{
	class Units
	{
	public:
		std::string length = "m";
		std::string mass = "kg";
		std::string time = "s";
		double characteristic_length = 1;

		std::string stress() const { return fmt::format("{}/({}*{}^2)", mass, length, time); }
		std::string density() const { return fmt::format("{}/{}^3", mass, length); }
		std::string velocity() const { return fmt::format("{}/{}", length, time); }
		std::string acceleration() const { return fmt::format("{}/{}^2", length, time); }
		std::string force() const { return fmt::format("{}*{}", mass, acceleration()); }
		std::string energy() const { return fmt::format("{}*{}^2/{}^2", mass, length, time); }
	};
} // namespace polyfem
