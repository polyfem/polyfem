#include "Units.hpp"

#include <polyfem/utils/Logger.hpp>

#include <units/units.hpp>

namespace polyfem
{
	void Units::init(const json &json)
	{
		const std::string json_length = json["length"];
		if (!json_length.empty())
			length_ = json_length;

		const std::string json_mass = json["mass"];
		if (!json_mass.empty())
			mass_ = json_mass;

		const std::string json_time = json["time"];
		if (!json_time.empty())
			time_ = json_time;

		if (json["characteristic_length"].is_number())
		{
			const double cl = json["characteristic_length"];
			if (cl > 0)
				characteristic_length_ = cl;
		}
	}

	double Units::convert(const json &val, const std::string &unit_type_s)
	{
		if (val.is_number())
			return val.get<double>();

		assert(val.is_object());

		return convert(val["value"].get<double>(), val["unit"].get<std::string>(), unit_type_s);
	}

	double Units::convert(const double val, const std::string &unit_s, const std::string &unit_type_s)
	{
		auto unit = units::unit_from_string(unit_s);
		auto unit_type = units::unit_from_string(unit_type_s);
		if (!unit.is_convertible(unit_type))
		{
			log_and_throw_error(fmt::format("Cannot convert {} to {}", units::to_string(unit), units::to_string(unit_type)));
		}

		return units::convert(val, unit, unit_type);
	}

} // namespace polyfem
