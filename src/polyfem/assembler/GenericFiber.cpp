#include "GenericFiber.hpp"

#include <polyfem/assembler/HGOFiber.hpp>
#include <polyfem/assembler/ActiveFiber.hpp>

namespace polyfem::assembler
{
	template <typename FiberModel>
	GenericFiber<FiberModel>::GenericFiber()
	{
	}

	template <typename FiberModel>
	void GenericFiber<FiberModel>::add_multimaterial(const int index, const json &params, const Units &units)
	{
		if (params.contains("fiber_direction"))
			fiber_direction_.add_multimaterial(index, params["fiber_direction"], units.length());
	}

	template <typename FiberModel>
	void GenericFiber<FiberModel>::set_size(const int size)
	{
		GenericElastic<FiberModel>::set_size(size);

		fiber_direction_.resize(size);
	}

	template <typename FiberModel>
	std::map<std::string, Assembler::ParamFunc> GenericFiber<FiberModel>::parameters() const
	{
		std::map<std::string, Assembler::ParamFunc> res;

		const auto &fiber_direction = this->fiber_direction_;

		res["fiber_direction_x"] = [&fiber_direction](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			Eigen::Vector3d tmp = fiber_direction(p, p, t, e);
			return tmp[0];
		};

		res["fiber_direction_y"] = [&fiber_direction](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			Eigen::Vector3d tmp = fiber_direction(p, p, t, e);
			return tmp[1];
		};

		if (this->size() == 3)
		{
			res["fiber_direction_z"] = [&fiber_direction](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
				Eigen::Vector3d tmp = fiber_direction(p, p, t, e);
				return tmp[2];
			};
		}

		return res;
	}

	template class GenericFiber<HGOFiber>;
	template class GenericFiber<ActiveFiber>;
} // namespace polyfem::assembler