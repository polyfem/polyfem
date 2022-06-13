#pragma once

#include <vector>
#include <string>
#include <filesystem>
// #include <>
namespace polyfem
{
	namespace Sanitization
	{
		void Sanitization::input_json_sanitization();
		void Sanitization::input_geom_sanitization();
		void Sanitization::input_simul_sanitization();
	} // namespace Sanitization
} // namespace polyfem